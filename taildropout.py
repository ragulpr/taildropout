import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


def get_params(p=0.5, lr=1e-5):
    """ Numerically solve integral equation int_0^1 S(x) dx = p
        with S survival function for exponential distribution.
        
        This is a very naive way of calculating it.
    """
    p = 1 - p  # Probability of dropout i.e prob. of zero
    from math import exp
    G = lambda a: a - a * exp(-1 / a)  # int_0^1 S(x) dx
    
    if (1-p)<0.01:
        lr = 1. # Since too slow otherwise.

    a = 0
    err = 2.
    # *Extremely* naive stepwise search from 0,0+lr,...
    while a < 100000:
        a = a + lr
        err_last = err
        err = (G(a) - p)**2
        if err > err_last:
            break
    return a

def ones_replaced(n, dim, val):
    """ List of n-1 ones and `val` at `dim`'th position.
    """
    ix = [1 for _ in range(n)]
    ix[dim] = val
    return ix


def _legacy_slice_zerofill(mask, dropout_dim, dropout_start):
    """ Helper for pytorch 0.2x, 0.3x compatibility.

        `.slice()` is not supported in pytorch <0.4

        In pytorch 0.4x we'd do:
        `mask.slice(dropout_dim,dropout_start).fill_(0)`

        But in pytorch 0.2 we do a dirty trick for dropout_dim<7 lol

    """
    if dropout_dim < -1:
        # to support negative indexing.
        dropout_dim = len(mask.shape) + dropout_dim

    if dropout_dim == -1:
        mask[..., dropout_start:] = 0
    elif dropout_dim == 0:
        mask[dropout_start:] = 0
    elif dropout_dim == 1:
        mask[:, dropout_start:] = 0
    elif dropout_dim == 2:
        mask[:, :, dropout_start:] = 0
    elif dropout_dim == 3:
        mask[:, :, :, dropout_start:] = 0
    elif dropout_dim == 4:
        mask[:, :, :, :, dropout_start:] = 0
    elif dropout_dim == 5:
        mask[:, :, :, :, :, dropout_start:] = 0
    elif dropout_dim == 6:
        mask[:, :, :, :, :, :, dropout_start:] = 0
    else:
        raise ValueError(
            'Expected dropout_dim = -1 or <7 but got ', dropout_dim)


class TailDropout(nn.Module):
    r"""During training, randomly zeroes last n columns.

        >> dropout = TailDropout()
        >> y = dropout(x)
        >> 
        >> dropout.eval()
        >> y = dropout(x)
        >> assert y.equal(x)
        >> 
        >> y = dropout(x,dropout_start = 10)
        >> assert y[:,10:].sum()==0
    """

    def __init__(self, p=0.5, batch_dim=0, dropout_dim=-1):
        super(TailDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))

        self.batch_dim = batch_dim
        self.dropout_dim = dropout_dim

        self.cdf = lambda x, scale: 1 - \
            torch.exp(-x / scale)  # expo distribution
        self.set_p(p)

    def set_p(self,p):
        self.p = p
        if p == 0 or p == 1:
            self.scale = None
        else:
            self.scale = get_params(p, lr=1e-5)

    def forward(self, input, dropout_start=None):
        n_batch = input.shape[self.batch_dim]
        n_features = input.shape[self.dropout_dim]

        if dropout_start is None:
            if self.training:
                if self.p == 0:
                    mode = 'straight-through'
                elif self.p == 1:
                    mode = 'zero'
                else:
                    mode = 'random'
            else:
                mode = 'straight-through'
        else:
            if dropout_start == n_features:
                mode = 'straight-through'
            elif dropout_start == 0:
                mode = 'zero'
            elif dropout_start > n_features:
                raise ValueError("dropout_start ({}) greater than n_features ({})".format(
                    dropout_start, n_features))
            else:
                mode = 'first_n'

        if mode == 'random':
            type_out = input.data.type() if isinstance(input, Variable) else input.type()

            n_dim = len(input.shape)
            # torch.linspace not cuda
            linspace = torch.arange(1, n_features + 1, 1).type(type_out)
            # resized [1,n_features] if input 2d
            linspace.resize_(ones_replaced(
                n_dim, self.dropout_dim, n_features))
            # self.scale*n_features faster than linspace/n_features
            prob = self.cdf(linspace, self.scale * n_features)

            if isinstance(input, Variable):
                # resized [n_batch,1] if input 2d
                uniform = input.data.new().resize_(
                    ones_replaced(n_dim, self.batch_dim, n_batch)).uniform_()
            else:
                # in pytorch 0.4 variable.new() works too.
                # resized [n_batch,1] if input 2d
                uniform = input.new().resize_(
                    ones_replaced(n_dim, self.batch_dim, n_batch)).uniform_()
            mask = prob < uniform         # 43% of cpu cumtime
            if isinstance(input, Variable):
                mask = Variable(mask)
            mask = mask.type(type_out)    # 30% of cpu cumtime
            return input * mask           # 23% of cpu cumtime # Note works due to broadcasting
            # Tempting to do masked_fill
            # But API unstable over cuda/cpu/pytorch 0.3/pytorch 0.4 
            # and probably be more memory costly.

        if mode == 'straight-through':
            return input
        if mode == 'first_n':
            mask = torch.ones_like(input)
            try:
                # Pytorch 0.4x
                mask.slice(self.dropout_dim, dropout_start).fill_(0)
            except:
                # Pytorch <0.4
                _legacy_slice_zerofill(mask, self.dropout_dim, dropout_start)

            return input * mask
        if mode == 'zero':
            return input * 0

        raise ValueError

    def __repr__(self):
        return self.__class__.__name__ +\
            '(p=' + str(self.p) + ', batch_dim=' + str(self.batch_dim) + \
            ', dropout_dim=' + str(self.dropout_dim) + ')'

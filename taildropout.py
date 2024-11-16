import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import Tensor
from typing import Union, List, Optional
from math import exp

def get_scale_param(p=0.5, tol=1e-6) -> float:
    """ Numerically solve integral equation int_0^1 S(x) dx = p
        with S survival function for exponential distribution.

        use a - a * exp(-1 / a) = int_0^1 S(x) dx = p
        <=> a = p/(1 - exp(-1/a)) fixed point form 
        and just iterate a=f(a) until convergence

        This is an arbitrary and naive way of calculating it.
    """
    assert p!=0 and p!=1

    p = 1 - p     # Convert to dropout prob
    a = 0.5/(1-p) # A good first guess especially for small (input) p
    err=1+tol
    while(err > tol):
        a = p/(1 - exp(-1/a))
        err = abs(p-(a - a * exp(-1 / a)))
    return a

def replace_w_ones_except(shape, dims):
    # List like `shape` with ones everywhere except at `dims`.
    newshape = [1 for _ in range(len(shape))]
    try:
        newshape[dims] = shape[dims]
    except:
        # dims iterable
        for j in dims:
            newshape[j] = shape[j]
    return newshape


class TailDropout(nn.Module):
    """During training, randomly zeroes last n-k columns using exponential dropout probability.
    
    This module implements a form of structured dropout where features are dropped out
    with increasing probability based on their position. This encourages the network
    to learn features in order of importance.
    
    Args:
        p: dropout probability (default: 0.5)
        batch_dim: dimension(s) to treat as batch (default: 0)
        dropout_dim: dimension to apply dropout to (default: -1)
    
    Example:
        >>> dropout = TailDropout()
        >>> x = torch.randn(20, 16)
        >>> y = dropout(x)
        >>> 
        >>> dropout.eval()
        >>> y = dropout(x)
        >>> assert y.equal(x)
        >>> 
        >>> y = dropout(x,dropout_start = 10)
        >>> assert y[:,10:].sum()==0
    """

    def __init__(self, 
                 p: float = 0.5, 
                 batch_dim: Union[int, List[int]] = 0, 
                 dropout_dim: int = -1) -> None:
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")

        is_overlap = False
        try:
            is_overlap = dropout_dim in batch_dim
        except:
            is_overlap = dropout_dim == batch_dim

        if is_overlap:
            raise ValueError(f"batch_dim ({batch_dim}) and dropout_dim {batch_dim} can't overlap")

        self.batch_dim = batch_dim
        self.dropout_dim = dropout_dim

        # exponential distribution
        self.cdf = lambda x, scale: 1 - torch.exp(-x / scale)
        self.set_p(p)

    def set_p(self, p)->None:
        self._p = p
        if p == 0 or p == 1:
            self.scale = None
        else:
            self.scale = get_scale_param(p)

    def forward(self, input: Tensor, dropout_start: Optional[int] = None) -> Tensor:
        n_features = input.shape[self.dropout_dim]

        if dropout_start is None:
            if self.training:
                if self._p == 0:
                    mode = 'straight-through'
                elif self._p == 1:
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
                raise ValueError(f"dropout_start ({dropout_start}) greater than n_features ({n_features})")
            else:
                mode = 'first_n'

        if mode == 'random':
            type_out = input.data.type() if isinstance(input, Variable) else input.type()

            n_dim = len(input.shape)
            # No cuda torch.linspace for old versions of pytorch.
            linspace = torch.arange(1, n_features + 1, 1).type(type_out)
            # resized [1,n_features] if input 2d, [1,1,..,n_features] if nd
            newshape = replace_w_ones_except(input.shape, self.dropout_dim)
            linspace.resize_(newshape)
            # self.scale*n_features faster than linspace/n_features
            prob = self.cdf(linspace, self.scale * n_features)

            # make [n_batch,1] noise if input 2d
            newshape = replace_w_ones_except(input.shape, self.batch_dim)
            if isinstance(input, Variable):
                uniform = input.data.new().resize_(newshape).uniform_()
            else:
                # in pytorch >0.4 variable.new() works too.
                uniform = input.new().resize_(newshape).uniform_()
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
            # Do mask[:, :, (...), :), :, dropout_start:] = 0 depending on dropout_dim
            slices = [slice(None)] * input.ndim  # Start with full slices for all dimensions
            slices[self.dropout_dim] = slice(dropout_start, None)  # Modify only the dropout_dim

            # Use the slice object to set values to zero
            mask[tuple(slices)] = 0

            return input * mask
        if mode == 'zero':
            return input * 0

        raise ValueError

    def extra_repr(self) -> str:
        return f'p={self._p}, batch_dim={self.batch_dim}, dropout_dim={self.dropout_dim}'

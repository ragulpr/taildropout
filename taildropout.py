import torch
import torch.nn as nn
from torch import Tensor
from typing import Union, List, Optional
from math import exp
import warnings

def get_scale_param(p, tol=1e-9) -> float:
    """ Numerically solve integral equation int_0^1 S(x) dx = p
        with S survival function for exponential distribution.

        use a - a * exp(-1 / a) = int_0^1 S(x) dx = p
        <=> a = p/(1 - exp(-1/a)) fixed point form 
        and just iterate a=g(a) until convergence

        This is a naive way of calculating it.
        But fast (nano-microseconds)
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
    newshape = [1]*len(shape)
    dims = [dims] if isinstance(dims, int) else dims
    for d in dims:
        newshape[d] = shape[d]
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
        >>> dropout.set_k(10)
        >>> y = dropout(x)
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
        self.register_buffer("_k", torch.tensor([0], dtype=torch.long), persistent=False)
        self.set_k(None)

    def set_p(self, p)->None:
        self._p = p
        if p == 0 or p == 1:
            self.scale = None
        else:
            self.scale = get_scale_param(p)

    def set_k(self, k:Optional[int]) :
        if k is None:
            # Least bad alt. but confusing as `[0,1][:-1] == [0]`
            # But we will use it as k=n_features (but dont want to set n_features at build time) 
            self._k[0] = -1
        else:
            assert k>=0, f"k={k} but expected k>=0 or k=None"
            self._k[0] = k

    def train(self, mode=True):
        if self._k != -1:
            warnings.warn("Calling .train() resets `self.k={self.k}` to None")
            self.set_k(None)
        return super().train(mode)

    def eval(self):
        if self._k != -1:
            warnings.warn("Calling .eval() resets `self.k={self.k}` to None")
            self.set_k(None)
        return super().eval()
    
    def forward(self, input: Tensor) -> Tensor:
        n_features = input.shape[self.dropout_dim]

        if self._k == -1:
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
            if self._k == n_features:
                mode = 'straight-through'
            elif self._k == 0:
                mode = 'zero'
            elif self._k > n_features:
                raise ValueError(f"TailDropout k ({self.k}) is greater than n_features ({n_features})")
            else:
                mode = 'first_k'

        if mode == 'random':
            type_out = input.dtype
            device = input.device

            linspace = torch.arange(1, n_features + 1, 1, device=device, dtype=type_out)
            prob_shape = replace_w_ones_except(input.shape, self.dropout_dim) #[1,1,..,n_features]
            linspace.resize_(prob_shape)
            # self.scale*n_features faster than linspace/n_features
            prob = self.cdf(linspace, self.scale * n_features)

            mask_shape = replace_w_ones_except(input.shape, self.batch_dim)
            uniform = torch.rand(mask_shape, device=device, dtype=type_out) # [n_batch,1,1] if input 3d
            mask = prob < uniform         # 43% of cpu cumtime                [n_batch,1,n_features] 
            mask = mask.type(type_out)    # 30% of cpu cumtime
            return input * mask           # 23% of cpu cumtime # Note works due to broadcasting
            # Similar performance / identical with torch.compile but doesn't propagate NaN:
            # inv_mask = prob >= uniform
            # return input.masked_fill(inv_mask, 0)

        if mode == 'straight-through':
            return input
        
        if mode == 'first_k':
            # Do mask[:, :, (...), :, k:] = 0 in choice of dropout_dim
            mask_shape = replace_w_ones_except(input.shape, self.dropout_dim)
            mask = input.new_ones(*mask_shape)
            slices = [slice(None)] * input.ndim

            slices[self.dropout_dim] = slice(self._k, None)
            mask[tuple(slices)] = 0

            return input * mask
        
        if mode == 'zero':
            return input * 0

        raise ValueError

    def extra_repr(self) -> str:
        return f'p={self._p}, batch_dim={self.batch_dim}, dropout_dim={self.dropout_dim}'

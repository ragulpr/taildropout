import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

def get_params(p = 0.5,lr = 1e-5):
    """ Numerically solve integral equation int_0^1 S(x) dx = p
        with S survival function for exponential distribution
    """
    p = 1-p # Probability of dropout i.e prob. of zero
    from math import exp
    G = lambda a : a - a*exp(-1/a)
    a = 0
    err = 2.
    while a<20:
        a = a+lr
        err_last = err
        err = (G(a)-p)**2
        if err>err_last:
            break
    return a

class ContiguousDropout(nn.Module):
    r"""During training, randomly zeroes last n columns.
        Half speed of linear-fused op.
        
        Assumes dropout dimension = -1
    """
    def __init__(self,p=0.5):
        super(ContiguousDropout, self).__init__()
        self.scale = get_params(p,lr = 1e-5)
        self.cdf = lambda x,scale : 1-torch.exp(-x/scale) # expo distribution
    
    def forward(self, input, dropout_start = None):
        if self.training and dropout_start is None:        
            n,k = input.shape[0],input.shape[-1]
            linspace = torch.linspace(1/k,(k-1)/k,k).type(input.type())            
            uniform = input.new(torch.Size([n,1])).uniform_()
            prob = self.cdf(linspace,self.scale)
            mask = prob<uniform
            return input*mask.type(input.type())      
            
        if dropout_start is not None:
            # Evaluation
            if dropout_start>=input.shape[-1]:
                # straight through
                return input
            
            mask = torch.zeros_like(input)
            if dropout_start>0:
                mask[...,:dropout_start] = 1.
            return input*mask
        
        if not self.training and dropout_start is None:
            # Evaluation : straight through
            return input

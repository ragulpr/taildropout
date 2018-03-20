import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class PartialLinear(nn.Linear):
    r"""A linear layer that only uses the first k input features.
    
    Equivalent to dropping out the last N-k input features.
    Equivalent to setting last N-k input features to zero.
    Equivalent to setting last N-k weight-columns to zero.
        
    Theoretically faster than dropout. 
    Can be sampled from discretized Beta to force inclusion probability towards equal.
    
    Examples::
        >>> linear = PartialLinear(3, 2)
        >>> input = Variable(torch.ones(2, 3))
        >>> input[:,2] = 1000
        >>> output = linear(input,2)
        >>> print(output)
    """

    def __init__(self, in_features, out_features, bias=True):
        super(PartialLinear, self).__init__(in_features, out_features, bias)
        
    def forward(self,input,k):
        if k>0:
            return F.linear(input[...,:k], self.weight[...,:k], self.bias)
        else:
            return F.linear(input*0., self.weight, self.bias)

class LinearWithContiguousDropout(nn.Module):
    r"""During training, randomly zeroes last n columns
    """
    def __init__(self, in_features,out_features,bias=True):
        super(LinearWithContiguousDropout, self).__init__()
        self.linear = PartialLinear(in_features,out_features,bias=True)
        self.in_features = in_features
    
    def forward(self, input,dropout_start = None):
        if self.training and dropout_start is None:
            dropout_start = torch.LongTensor(1).random_(self.in_features)[0]
        elif dropout_start is None:
            dropout_start = self.in_features
        return self.linear(input,dropout_start)

class LinearWithContiguousDropoutMasked(nn.Module):
    r"""During training, randomly zeroes last n columns
    """
    def __init__(self, in_features,out_features,bias=True):
        super(LinearWithContiguousDropoutMasked, self).__init__()
        self.linear = PartialLinear(in_features,out_features,bias=True)
        self.in_features = in_features
    
    def forward(self, input, dropout_start = None):
        if self.training and dropout_start is None:
            noise = input.new().resize_as_(input).bernoulli_(1./self.in_features).cumsum(-1)==0
            noise = noise.type(input.type())
            return self.linear(noise*input,self.in_features)
        
        if dropout_start is None:
            dropout_start = self.in_features + 100
        return self.linear(input,dropout_start)

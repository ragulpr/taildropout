import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class ContiguousDropout(nn.Module):
    r"""During training, randomly zeroes last n columns.
        Half speed of linear-fused op.
        
        Assumes dropout dimension = -1
    """
    def __init__(self):
        super(ContiguousDropout, self).__init__()
    
    def forward(self, input, dropout_start = None):
        if self.training and dropout_start is None:        
            in_features = input.shape[-1]
            # 1.54 magic number -> approx 0.5 dropout prob.
            p = min(in_features,1.54)*(1./in_features)
            # TODO speed up mask-creation.
            mask = input.new().resize_as_(input).bernoulli_(p).cumsum(-1)==0
            mask = mask.type(input.type())
            return input*mask
        
        if dropout_start is not None:
            # Evaluation
            mask = torch.zeros_like(input)
            if dropout_start>0:
                mask[...,:dropout_start] = 1.
            return input*mask
        
        if not self.training and dropout_start is None:
            # Evaluation : straight through
            return input

class ContiguousDropout2(nn.Module):
    r"""During training, randomly zeroes last n columns.
        Half speed of linear-fused op.
        
        Assumes dropout dimension = -1
    """
    def __init__(self):
        super(ContiguousDropout2, self).__init__()
    
    def forward(self, input, dropout_start = None):
        if self.training and dropout_start is None:        
            n,k = input.shape[0],input.shape[-1]
            sample = torch.distributions.Beta(1,2).rsample(torch.Size([n,1]))
            p = torch.linspace(1/k,(k-1)/k,k)
            mask = sample>=p*.605
            mask = Variable(mask.type(input.type()))
            return input*mask
        
        if dropout_start is not None:
            # Evaluation
            mask = torch.zeros_like(input)
            if dropout_start>0:
                mask[...,:dropout_start] = 1.
            return input*mask
        
        if not self.training and dropout_start is None:
            # Evaluation : straight through
            return input

class ContiguousDropout3(nn.Module):
    r"""During training, randomly zeroes last n columns.
        Half speed of linear-fused op.
        
        Assumes dropout dimension = -1
    """
    def __init__(self):
        super(ContiguousDropout2, self).__init__()
    
    def forward(self, input, dropout_start = None):
        if self.training and dropout_start is None:        
            n,k = input.shape[0],input.shape[-1]
            """
                Each row choses the ~Beta(a,b)'th index
                Verbose due to storage/type-match.
            """
            tmp = input.new(1)
            sample = torch.distributions.Beta(tmp+1,tmp+2).rsample(torch.Size([n]))
            linspace = torch.linspace(1/k,(k-1)/k,k)
            if isinstance(input, Variable):
                p = input.new().new(torch.Size([k])).copy_(Variable(linspace.type(input.type())))
            else:
                p = input.new().new(torch.Size([k])).copy_(linspace.type(input.type()))
            mask = sample>=p*.605
            return input*mask.type(input.type())      
            
        if dropout_start is not None:
            # Evaluation
            mask = torch.zeros_like(input)
            if dropout_start>0:
                mask[...,:dropout_start] = 1.
            return input*mask
        
        if not self.training and dropout_start is None:
            # Evaluation : straight through
            return input
    
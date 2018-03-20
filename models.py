import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from dropout import *
from linearwithdropout import *


class SequentialDropoutBatchwise(nn.Module):
    r"""During training, randomly zeroes last n columns
    """
    def __init__(self):
        super(SequentialDropoutBatchwise, self).__init__()
        self.linear1 = nn.Linear(1,n_hidden,bias=False)
        self.linear2 = LinearWithContiguousDropout(n_hidden,n_hidden)
        self.linear3 = LinearWithContiguousDropout(n_hidden,n_hidden)
        self.linear_out = LinearWithContiguousDropout(n_hidden,1)
    def forward(self, x, dropout_start = None):
        x = self.linear1(x).tanh()
        x = self.linear2(x,dropout_start).tanh()
        x = self.linear3(x,dropout_start).tanh()
        x = self.linear_out(x,dropout_start)
        return x

class SequentialDropout(nn.Module):
    r"""During training, randomly zeroes last n columns
    """
    def __init__(self):
        super(SequentialDropout, self).__init__()
        self.linear1 = nn.Linear(1,n_hidden,bias=False)
        self.linear2 = LinearWithContiguousDropoutMasked(n_hidden,n_hidden)
        self.linear3 = LinearWithContiguousDropoutMasked(n_hidden,n_hidden)
        self.linear_out = LinearWithContiguousDropoutMasked(n_hidden,1)
    def forward(self, x, dropout_start = None):
        x = self.linear1(x).tanh()
        x = self.linear2(x,dropout_start).tanh()
        x = self.linear3(x,dropout_start).tanh()
        x = self.linear_out(x,dropout_start)
        return x

class Deterministic(nn.Module):
    def __init__(self):
        super(Deterministic, self).__init__()
        self.linear1 = nn.Linear(1,n_hidden,bias=False)
        self.linear2 = LinearWithContiguousDropout(n_hidden,n_hidden)
        self.linear3 = LinearWithContiguousDropout(n_hidden,n_hidden)
        self.linear_out = LinearWithContiguousDropout(n_hidden,1)
    def forward(self, x, dropout_start = None):
        if dropout_start is None:
            dropout_start = 100000
        x = self.linear1(x).tanh()
        x = self.linear2(x,dropout_start).tanh()
        x = self.linear3(x,dropout_start).tanh()
        x = self.linear_out(x,dropout_start)
        return x

class RegularDropout(nn.Module):
    r"""
    """
    def __init__(self):
        super(RegularDropout, self).__init__()
        self.linear1 = nn.Linear(1,n_hidden,bias=False)
        self.linear2 = LinearWithContiguousDropout(n_hidden,n_hidden)
        self.linear3 = LinearWithContiguousDropout(n_hidden,n_hidden)
        self.linear_out = LinearWithContiguousDropout(n_hidden,1)
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, x, dropout_start = None):
        if dropout_start is None:
            dropout_start = 100000
        x = self.dropout(self.linear1(x).tanh())
        x = self.dropout(self.linear2(x,dropout_start).tanh())
        x = self.dropout(self.linear3(x,dropout_start).tanh())
        x = self.linear_out(x,dropout_start)
        return x

class RegularDropout2(nn.Module):
    r""" Control using regular nn.Linear layer. Just for assert of equality.
    """
    def __init__(self):
        super(RegularDropout2, self).__init__()
        self.linear1 = nn.Linear(1,n_hidden,bias=False)
        self.linear2 = nn.Linear(n_hidden,n_hidden)
        self.linear3 = nn.Linear(n_hidden,n_hidden)
        self.linear_out = nn.Linear(n_hidden,1)
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, x, dropout_start = None):
        x = self.dropout(self.linear1(x).tanh())
        x = self.dropout(self.linear2(x).tanh())
        x = self.dropout(self.linear3(x).tanh())
        x = self.linear_out(x)
        return x

class SequentialDropout2(nn.Module):
    r""" Control using regular nn.Linear layer. Just for assert of equality.
    """
    def __init__(self):
        super(SequentialDropout2, self).__init__()
        self.linear1 = nn.Linear(1,n_hidden,bias=False)
        self.linear2 = nn.Linear(n_hidden,n_hidden)
        self.linear3 = nn.Linear(n_hidden,n_hidden)
        self.linear_out = nn.Linear(n_hidden,1)
        self.dropout = ContiguousDropout()
    def forward(self, x, dropout_start = None):

        x = self.dropout(self.linear1(x).tanh(),dropout_start)
        x = self.dropout(self.linear2(x).tanh(),dropout_start)
        x = self.dropout(self.linear3(x).tanh(),dropout_start)
        x = self.linear_out(x)
        return x

class SequentialDropout3(nn.Module):
    r""" Control using regular nn.Linear layer. Just for assert of equality.
    """
    def __init__(self):
        super(SequentialDropout3, self).__init__()
        self.linear1 = nn.Linear(1,n_hidden,bias=False)
        self.linear2 = nn.Linear(n_hidden,n_hidden)
        self.linear3 = nn.Linear(n_hidden,n_hidden)
        self.linear_out = nn.Linear(n_hidden,1)
        self.dropout = ContiguousDropout2()
    def forward(self, x, dropout_start = None):

        x = self.dropout(self.linear1(x).tanh(),dropout_start)
        x = self.dropout(self.linear2(x).tanh(),dropout_start)
        x = self.dropout(self.linear3(x).tanh(),dropout_start)
        x = self.linear_out(x)
        return x
n_hidden= 50

def test_models():
    # All vals should be equal at inference mode
    n_hidden= 50
    vals = []
    for Model in [SequentialDropout3,SequentialDropoutBatchwise,SequentialDropout,Deterministic,RegularDropout,RegularDropout2,SequentialDropout2]:
        torch.manual_seed(1)
        model = Model()
        model.eval()
        vals.append(model(Variable(torch.randn(1000, 1))).sum().data[0])    
    assert len(set(vals))==1

test_models()


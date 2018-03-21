import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from dropout import ContiguousDropout

def test_ContiguousDropout():
    n = 10
    k = 10
    dropout = ContiguousDropout()

    x = Variable(torch.ones(n,k))    
    y = dropout(x,2)
    # all but first 2 cols zero
    assert y[:,2:].sum()==0
    # first 2 cols ones
    assert torch.mean(y[:,:2])==1
    dropout.eval()
    y = dropout(x)
    # all columns exactly one
    assert y.mean()==1.
    
    # Test routes
    y_eval = dropout(x)
    y_eval2 = dropout(x,5)
    dropout.train()
    y_train = dropout(x,1000)
    y_train2 = dropout(x,5)
    assert (y_eval == y_train).all()
    assert (y_eval2 == y_train2).all()
    
    # Test that dropout probability is correct
    torch.manual_seed(1)
    n = 10000
    k = 100
    for p in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        dropout = ContiguousDropout(p)
        y = dropout(Variable(torch.ones(n,k)))    
        assert (1-y.mean()-p).abs()<1e-2,[1-y.mean().data.numpy()[0],p]
    
    
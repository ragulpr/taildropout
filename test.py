import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from dropout_ import ContiguousDropout

def test_ContiguousDropout():
    n = 10
    k = 10
    dropout = ContiguousDropout()

    # test inputs
    assert dropout(torch.ones(n,k)).shape == (n,k) # tensor
    assert dropout(Variable(torch.ones(n,k))).shape == (n,k) # variable

    assert dropout(torch.ones(1,k)).shape == (1,k),dropout(torch.ones(1,k)).shape # 
    assert dropout(torch.ones(1,1)).shape == (1,1) # 

    dropout.eval()
    assert dropout(torch.ones(1,k)).shape == (1,k) # 
    assert dropout(torch.ones(1,1)).shape == (1,1) # 
    dropout.train()

    x = Variable(torch.ones(n,k))    
    y = dropout(x,2)
    # all but first 2 cols zero
    assert (y[:,2:].sum()==0).all()
    # first 2 cols ones
    assert (torch.mean(y[:,:2])==1).all()
    dropout.eval()
    y = dropout(x)
    # all columns exactly one
    assert (y.mean()==1.).all()
    
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
    n = 100000
    for k in [10,50,100,1000]:
        epsilon = 2e-2
        if k ==10:
            epsilon = 5e-2
        x = Variable(torch.ones(n,k))
        for p in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
            dropout = ContiguousDropout(p)
            y = dropout(x)
            err = (1-y.mean()-p).abs()
            print(k,p,1-y.mean().cpu().data.numpy(),err.cpu().data.numpy())
            assert err<epsilon,[k,p,1-y.mean().cpu().data.numpy(),err.cpu().data.numpy()]
    

test_ContiguousDropout()
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print('GPU;')
    test_ContiguousDropout()
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from dropout import ContiguousDropout

def test_expected_mask():
    def nd_asserter(dropout, x):
        # Assert shapes
        dropout.train()
        assert dropout(x).shape == x.shape 
        assert dropout(x, 2).shape == x.shape 
        assert dropout(x.data).shape == x.shape # tensor
        dropout.eval()
        assert dropout(x).shape == x.shape 
        assert dropout(x, 2).shape == x.shape 
        assert dropout(x.data).shape == x.shape # tensor

        # Assert values
        dropout.train()
        y = dropout(x, 2).squeeze()
        # all but first 2 cols zero
        assert (y[:, 2:].sum() == 0).all(), y
        # first 2 cols ones
        assert (torch.mean(y[:, :2]) == 1).all()
        dropout.eval()
        y = dropout(x)
        # all columns exactly one
        assert (y.mean() == 1.).all()

        # Test routes when dropout_start is/not None
        y_eval = dropout(x)
        y_eval2 = dropout(x, 5)
        dropout.train()
        y_train = dropout(x, k)
        y_train2 = dropout(x, 5)
        assert (y_eval == y_train).all()
        assert (y_eval2 == y_train2).all()

    n = 11
    k = 13
    nd_asserter(dropout=ContiguousDropout(),
                x=Variable(torch.ones(n, k)))
    nd_asserter(dropout=ContiguousDropout(batch_dim=1),
                x=Variable(torch.ones(1, n, k)))
    nd_asserter(dropout=ContiguousDropout(batch_dim=1),
                x=Variable(torch.ones(1, n, 1, k)))
    nd_asserter(dropout=ContiguousDropout(batch_dim=0, dropout_dim=-1),
                x=Variable(torch.ones(n, 1, k)))

    nd_asserter(dropout=ContiguousDropout(batch_dim=1, dropout_dim=-2),
                x=Variable(torch.ones(1, n, 1, k, 1)))
    nd_asserter(dropout=ContiguousDropout(batch_dim=1, dropout_dim=3),
                x=Variable(torch.ones(1, n, 1, k, 1)))

    # Test 0/1 probability
    nd_asserter(dropout=ContiguousDropout(0),
                x=Variable(torch.ones(n, k)))
    nd_asserter(dropout=ContiguousDropout(1),
                x=Variable(torch.ones(n, k)))

def test_grad():
    n = 2
    k = 5
    x = Variable(torch.ones(n, 1, 2, 3, k), requires_grad=True)

    for dropout in [ContiguousDropout(),
                    ContiguousDropout(0),
                    ContiguousDropout(1),
                    ContiguousDropout(dropout_dim=4)]:
        # Deterministic
        y = dropout(x, 2)
        y.sum().backward()
        assert x.grad.data.equal(y.data)

        x.grad.data.zero_()
        y = dropout(x)
        y.sum().backward()
        assert x.grad.data.equal(y.data)
        x.grad.data.zero_()


def test_dropoutprob():
    # Test that dropout probability is correct
    torch.manual_seed(1)
    n = 100000
    for k in [10, 50, 100, 1000]:
        epsilon = 2e-2
        if k == 10:
            epsilon = 5e-2
        x = Variable(torch.ones(n, k))

        for p in [0,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1.]:
            dropout = ContiguousDropout(p)
            y = dropout(x)
            err = (1 - y.mean() - p).abs()
            print(k, p, 1 - y.mean().cpu().data.numpy(), err.cpu().data.numpy())
            assert err < epsilon, [
                k, p, 1 - y.mean().cpu().data.numpy(), err.cpu().data.numpy()]

print('CPU;')
print('test_expected_mask')
test_expected_mask()
print('test_grad')
test_grad()
print('test_dropoutprob')
test_dropoutprob()
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print('GPU;')
    print('test_expected_mask')
    test_expected_mask()
    print('test_grad')
    test_grad()
    print('test_dropoutprob')
    test_dropoutprob()




import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from taildropout import TailDropout, _legacy_slice_zerofill


def test_expected_mask():
    def test_routes(dropout, input_shape, requires_grad=False):
        x = Variable(torch.ones(input_shape), requires_grad=requires_grad)
        if torch.cuda.is_available():
            x = x.cuda()
        # Assert shapes
        dropout.train()
        assert dropout(x).shape == x.shape
        assert dropout(x, 2).shape == x.shape
        dropout.eval()
        assert dropout(x).shape == x.shape
        assert dropout(x, 2).shape == x.shape

        # Assert shapes/forward for Tensor
        dropout.train()
        assert dropout(x.data).shape == x.shape
        assert dropout(x.data, 2).shape == x.shape
        dropout.eval()
        assert dropout(x.data).shape == x.shape
        assert dropout(x.data, 2).shape == x.shape

        # Test routes when dropout_start is/not None
        dropout.eval()
        y_eval = dropout(x)
        y_eval2 = dropout(x, 5)
        dropout.train()
        y_train = dropout(x, k)
        y_train2 = dropout(x, 5)
        assert (y_eval == y_train).all()
        assert (y_eval2 == y_train2).all()

        dropout.eval()
        y = dropout(x)
        # all columns exactly one
        assert (y.mean() == 1.).all()

    def test_values_dd_last(dropout, input_shape):
        # Assumes dropout dimension is the last dimension.
        x = Variable(torch.ones(input_shape))
        # Assert values
        dropout.train()
        y = dropout(x, 2).squeeze()
        # all but first 2 values zero
        assert (y[..., 2:].sum() == 0).all(), y
        # first 2 values ones
        assert (torch.mean(y[..., :2]) == 1).all()

    n = 5
    k = 7
    test_routes(dropout=TailDropout(), input_shape=(n, k))
    test_routes(dropout=TailDropout(batch_dim=1), input_shape=(1, n, k))
    test_routes(dropout=TailDropout(batch_dim=1), input_shape=(1, n, 1, k))
    test_routes(dropout=TailDropout(batch_dim=0, dropout_dim=-1), input_shape=(n, 1, k))
    test_routes(dropout=TailDropout(batch_dim=1, dropout_dim=-2), input_shape=(1, n, 1, k, 1))
    test_routes(dropout=TailDropout(batch_dim=1, dropout_dim=3), input_shape=(1, n, 1, k, 1))
    test_routes(dropout=TailDropout(batch_dim=1), input_shape=(1, n, k))

    test_values_dd_last(dropout=TailDropout(), input_shape=(n, k))
    test_values_dd_last(dropout=TailDropout(), input_shape=(n, 1, k))
    test_values_dd_last(dropout=TailDropout(), input_shape=(n, n, k))

    test_values_dd_last(dropout=TailDropout(dropout_dim = 1), input_shape=(n, k))
    test_values_dd_last(dropout=TailDropout(dropout_dim = 2), input_shape=(n, 1, k))
    test_values_dd_last(dropout=TailDropout(dropout_dim = 2), input_shape=(n, n, k))

    test_values_dd_last(dropout=TailDropout(batch_dim = 1), input_shape=(n, 1, k))
    test_values_dd_last(dropout=TailDropout(batch_dim = 1), input_shape=(n, n, k))
    test_values_dd_last(dropout=TailDropout(batch_dim = [0,1]), input_shape=(n, n, k))
    test_values_dd_last(dropout=TailDropout(batch_dim = [1,0]), input_shape=(n, n, k))

    # Test 0/1 probability
    test_routes(dropout=TailDropout(0), input_shape=(n, k))
    test_routes(dropout=TailDropout(1), input_shape=(n, k))

    # Variable with grad
    test_routes(dropout=TailDropout(), input_shape=(n, k), requires_grad=True)



def test_multiple_batch_dim():
    x = torch.ones(100,100,10)
    if torch.cuda.is_available():
        x = x.cuda()
    
    y = TailDropout(batch_dim = 0)(x).sum(-1).sum(0)
    # Mask[i,a] == Mask[i,b] for all i, a, b
    assert all((y==y[0]))

    y = TailDropout(batch_dim = [0,1])(x).sum(-1).sum(0)
    # Mask[i,a] probably different from Mask[i,b] for some i, a, b
    assert not all((y==y[0]))


def test_grad():
    n = 2
    k = 5
    x = Variable(torch.ones(n, 1, 2, 3, k), requires_grad=True)

    for dropout in [TailDropout(),
                    TailDropout(0),
                    TailDropout(1),
                    TailDropout(dropout_dim=4)]:
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
        x = Variable(torch.ones(n,2, k))

        print('K', '\t', 'p', '\t', 'observed_p', '\t', 'err')
        for p in [0, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]:
            dropout = TailDropout(p,batch_dim = [0,1])
            y = dropout(x)
            observed_p = (1 - y).mean()
            err = (observed_p - p).abs()
            print(k, '\t', p, '\t', observed_p.cpu().data.numpy(),
                  '\t', err.cpu().data.numpy())
            assert (err < epsilon).all()


def test_legacy_slice_zerofill():
    k = 100
    mask1 = torch.randn([10, k])
    mask2 = mask1.clone()
    dropout_dim = 0
    for dropout_start in range(k):
        mask1.slice(dropout_dim, dropout_start).fill_(0)
        _legacy_slice_zerofill(mask2, dropout_dim, dropout_start)
        assert mask1.equal(mask2)

print('torch version ', torch.__version__)
if torch.__version__[:3] < '0.4':
    print('test_legacy_slice_zerofill')
    test_legacy_slice_zerofill()

print('CPU;')
print('test_expected_mask')
test_expected_mask()
print('test_multiple_batch_dim')
test_multiple_batch_dim()
print('test_grad')
test_grad()
print('test_dropoutprob')
test_dropoutprob()
if torch.cuda.is_available():
    print('GPU;')
    print('test_expected_mask')
    test_expected_mask()
    print('test_multiple_batch_dim')
    test_multiple_batch_dim()
    print('test_grad')
    test_grad()
    print('test_dropoutprob')

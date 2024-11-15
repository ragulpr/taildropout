import torch
import torch.nn as nn
import torch.nn.functional as F
from taildropout import TailDropout

def test_expected_mask():
    def test_routes(dropout, input_shape, requires_grad=False):
        x = torch.ones(input_shape, requires_grad=requires_grad)
        if torch.cuda.is_available():
            x = x.cuda()
            
        # Assert shapes
        dropout.train()
        assert dropout(x).shape == x.shape
        assert dropout(x, 2).shape == x.shape
        dropout.eval()
        assert dropout(x).shape == x.shape
        assert dropout(x, 2).shape == x.shape

        # Assert shapes/forward for plain tensor
        dropout.train()
        assert dropout(x.detach()).shape == x.shape
        assert dropout(x.detach(), 2).shape == x.shape
        dropout.eval()
        assert dropout(x.detach()).shape == x.shape
        assert dropout(x.detach(), 2).shape == x.shape

        # Test routes when dropout_start is/not None
        dropout.eval()
        y_eval = dropout(x)
        y_eval2 = dropout(x, 5)
        dropout.train()
        y_train = dropout(x, k)
        y_train2 = dropout(x, 5)
        assert y_eval.equal(y_train)
        assert y_eval2.equal(y_train2)

        dropout.eval()
        y = dropout(x)
        # all columns exactly one
        assert y.mean().allclose(torch.tensor(1.))

    def test_values_dd_last(dropout, input_shape):
        # Assumes dropout dimension is the last dimension.
        x = torch.ones(input_shape)
        # Assert values
        dropout.train()
        y = dropout(x, 2).squeeze()
        # all but first 2 values zero
        assert torch.all(y[..., 2:].sum() == 0), y
        # first 2 values ones
        assert torch.mean(y[..., :2]).allclose(torch.tensor(1.))

    n = 5
    k = 7
    test_routes(dropout=TailDropout(), input_shape=(n, k))
    test_routes(dropout=TailDropout(batch_dim=1), input_shape=(1, n, k))
    test_routes(dropout=TailDropout(batch_dim=1), input_shape=(1, n, 1, k))
    test_routes(dropout=TailDropout(batch_dim=0, dropout_dim=-1), input_shape=(n, 1, k))  # noqa
    test_routes(dropout=TailDropout(batch_dim=1, dropout_dim=-2), input_shape=(1, n, 1, k, 1))  # noqa
    test_routes(dropout=TailDropout(batch_dim=1, dropout_dim=3), input_shape=(1, n, 1, k, 1))  # noqa
    test_routes(dropout=TailDropout(batch_dim=1), input_shape=(1, n, k))  # noqa

    test_values_dd_last(dropout=TailDropout(), input_shape=(n, k))  # noqa
    test_values_dd_last(dropout=TailDropout(), input_shape=(n, 1, k))  # noqa
    test_values_dd_last(dropout=TailDropout(), input_shape=(n, n, k))  # noqa

    test_values_dd_last(dropout=TailDropout(dropout_dim=1), input_shape=(n, k))
    test_values_dd_last(dropout=TailDropout(dropout_dim=2), input_shape=(n, 1, k))  # noqa
    test_values_dd_last(dropout=TailDropout(dropout_dim=2), input_shape=(n, n, k))  # noqa

    test_values_dd_last(dropout=TailDropout(batch_dim=1), input_shape=(n, 1, k))  # noqa
    test_values_dd_last(dropout=TailDropout(batch_dim=1), input_shape=(n, n, k))  # noqa
    test_values_dd_last(dropout=TailDropout(batch_dim=[0, 1]), input_shape=(n, n, k))  # noqa
    test_values_dd_last(dropout=TailDropout(batch_dim=[1, 0]), input_shape=(n, n, k))  # noqa

    # Test 0/1 probability
    test_routes(dropout=TailDropout(0), input_shape=(n, k))
    test_routes(dropout=TailDropout(1), input_shape=(n, k))

    # Variable with grad
    test_routes(dropout=TailDropout(), input_shape=(n, k), requires_grad=True)


def test_multiple_batch_dim():
    x = torch.ones(100, 100, 10)
    if torch.cuda.is_available():
        x = x.cuda()

    y = TailDropout(batch_dim=0)(x).sum(-1).sum(0)
    # Mask[i,a] == Mask[i,b] for all i, a, b
    assert all((y == y[0]))

    y = TailDropout(batch_dim=[0, 1])(x).sum(-1).sum(0)
    # Mask[i,a] probably different from Mask[i,b] for some i, a, b
    assert not all((y == y[0]))


def test_grad():
    n = 2
    k = 5
    x = torch.ones(n, 1, 2, 3, k, requires_grad=True)


    for dropout in [TailDropout(),
                    TailDropout(0),
                    TailDropout(1),
                    TailDropout(dropout_dim=4)]:
        # Deterministic
        y = dropout(x, 2)
        y.sum().backward()
        assert x.grad.detach().equal(y.detach())

        x.grad = None  # Modern way to zero gradients
        y = dropout(x)
        y.sum().backward()
        assert x.grad.detach().equal(y.detach())
        x.grad = None

def test_dropoutprob():
    # Test that dropout probability is correct
    torch.manual_seed(1)
    n = 100000
    for k in [10, 50, 100, 1000]:
        epsilon = 2e-2
        if k == 10:
            epsilon = 5e-2
        x = torch.ones(n, 2, k)

        print('K', '\t', 'p', '\t', 'observed_p', '\t', 'err')
        for p in [0, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]:
            dropout = TailDropout(p, batch_dim=[0, 1])
            y = dropout(x)
            observed_p = (1 - y).mean()
            err = abs(observed_p - p)
            print(f'{k}\t{p:.4f}\t{observed_p.item():.4f}\t{err.item():.4f}')
            assert err < epsilon


def test_first_n():
        x  = torch.randn([2,3,4,10,5])
        dropout_start = 6
        expected = x.clone()
        expected[:, :, :, dropout_start:] = 0
        actual =  TailDropout(dropout_dim=3)(x, dropout_start=dropout_start)
        assert actual.equal(expected)

print(f'torch version {torch.__version__}')
print(f'torch.cuda.is_available():{torch.cuda.is_available()}')

# print('CPU;')
# print('test_expected_mask')
# test_expected_mask()
# print('test_multiple_batch_dim')
# test_multiple_batch_dim()
# print('test_grad')
# test_grad()
# print('test_dropoutprob')
# test_dropoutprob()
# if torch.cuda.is_available():
#     print('GPU;')
#     print('test_expected_mask')
#     test_expected_mask()
#     print('test_multiple_batch_dim')
#     test_multiple_batch_dim()
#     print('test_grad')
#     test_grad()
#     print('test_dropoutprob')

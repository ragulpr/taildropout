from math import exp
import torch
from taildropout import TailDropout, get_scale_param

def test_expected_mask():
    def test_routes(dropout, input_shape, requires_grad=False):
        x = torch.ones(input_shape, requires_grad=requires_grad)
        if torch.cuda.is_available():
            x = x.cuda()
            
        # Assert shapes
        dropout.train()
        assert dropout(x).shape == x.shape
        dropout.set_k(2)
        assert dropout(x).shape == x.shape
        dropout.eval()
        assert dropout(x).shape == x.shape
        dropout.set_k(2)
        assert dropout(x).shape == x.shape


        # Test values in train, eval, prune mode
        dropout.eval()
        y_all_eval = dropout(x)
        dropout.set_k(2)
        y_k_eval = dropout(x)
        dropout.train()
        dropout.set_k(f)
        y_all_train = dropout(x)
        dropout.set_k(2)

        y_k_train = dropout(x)
        torch.testing.assert_close(y_all_eval, y_all_train)
        torch.testing.assert_close(y_k_eval, y_k_train)

        # all columns exactly one
        assert y_all_eval.mean().allclose(torch.tensor(1.))
        assert y_k_eval.mean().allclose(torch.tensor(2/f))

        if dropout.dropout_dim==-1 or dropout.dropout_dim == len(input_shape):
            # Assumes dropout dimension is the last dimension.
            x = torch.randn(input_shape)
            # Assert values
            dropout.set_k(2)
            y = dropout(x)
            torch.testing.assert_close(y[..., 2:], torch.zeros_like(y[..., 2:]))
            torch.testing.assert_close(y[..., :2], x[..., :2])

    n = 5
    f = 7

    test_routes(dropout=TailDropout(), input_shape=(n, f))  # noqa
    test_routes(dropout=TailDropout(), input_shape=(n, 1, f))  # noqa
    test_routes(dropout=TailDropout(), input_shape=(n, n, f))  # noqa

    test_routes(dropout=TailDropout(dropout_dim=1), input_shape=(n, f))
    test_routes(dropout=TailDropout(dropout_dim=2), input_shape=(n, 1, f))  # noqa
    test_routes(dropout=TailDropout(dropout_dim=2), input_shape=(n, n, f))  # noqa

    test_routes(dropout=TailDropout(batch_dim=0,  dropout_dim=-1), input_shape=(n, 1, f))  # noqa

    test_routes(dropout=TailDropout(batch_dim=1), input_shape=(1, n, 1, f)) # noqa
    test_routes(dropout=TailDropout(batch_dim=1), input_shape=(1, n, f))  # noqa
    test_routes(dropout=TailDropout(batch_dim=1), input_shape=(n, 1, f))  # noqa
    test_routes(dropout=TailDropout(batch_dim=1), input_shape=(n, n, f))  # noqa
    test_routes(dropout=TailDropout(batch_dim=1, dropout_dim=-2), input_shape=(1, n, 1, f, 1))  # noqa
    test_routes(dropout=TailDropout(batch_dim=1, dropout_dim=3), input_shape=(1, n, 1, f, 1))  # noqa


    test_routes(dropout=TailDropout(batch_dim=[0, 1]), input_shape=(n, n, f))  # noqa
    test_routes(dropout=TailDropout(batch_dim=[1, 0]), input_shape=(n, n, f))  # noqa


    # Test 0/1 probability
    x = torch.ones([n,f])
    if torch.cuda.is_available():
        x = x.cuda()
    torch.testing.assert_close(TailDropout(0)(x),x)
    torch.testing.assert_close(TailDropout(1)(x),torch.zeros_like(x))

    # Variable with grad
    test_routes(dropout=TailDropout(), input_shape=(n, f), requires_grad=True)


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
        dropout.set_k(2)
        y = dropout(x)
        y.sum().backward()
        assert x.grad.detach().equal(y.detach())

        # Random
        dropout.train()
        x.grad = None
        y = dropout(x)
        y.sum().backward()
        assert x.grad.detach().equal(y.detach())
        x.grad = None

def test_get_scale_param():
    tol=1e-10
    for p_expected in [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        a = get_scale_param(p_expected,tol)
        p_actual = a - a * exp(-1 / a)  # int_0^1 S(x) dx
        assert abs(p_expected-(1-p_actual))<tol

def test_dropoutprob():
    # Integration test that dropout probability is correct up to errors from discretization.
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


def test_first_k():
        x  = torch.randn([2,3,4,10,5])
        dropout_start = 6
        expected = x.clone()
        expected[:, :, :, dropout_start:] = 0
        dropout=TailDropout(dropout_dim=3)
        dropout.set_k(dropout_start)
        actual =  dropout(x)
        assert actual.equal(expected)

print(f'torch version {torch.__version__}')
print(f'torch.cuda.is_available():{torch.cuda.is_available()}')

# print('test_expected_mask',test_expected_mask())
# print('test_multiple_batch_dim',test_multiple_batch_dim())
# print('test_grad',test_grad())
# print('test_dropoutprob',test_dropoutprob())
# print('test_first_k',test_first_k())

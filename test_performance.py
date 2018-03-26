import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from dropout import ContiguousDropout

import argparse
# python -m cProfile -s cumtime test_performance.py --repeats  10
parser = argparse.ArgumentParser(description='')
parser.add_argument('--repeats', type=int, default=1000000, metavar='N')
parser.add_argument('--n-features', type=int, default=1000, metavar='N')
parser.add_argument('--batch-size', type=int, default=1000, metavar='N')
parser.add_argument('--no-cuda', action='store_true',
                    default=False, help='disables CUDA training')
args = parser.parse_args()

print('args.no_cuda: ', args.no_cuda)
args.cuda = not args.no_cuda and torch.cuda.is_available()
print('GPU: ', args.cuda)
print(torch.__version__)
if args.cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

import time
import math


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def cdropout_forward(eval_mode = False):
    n = args.batch_size
    k = args.n_features
    dropout = ContiguousDropout()
    y = Variable(torch.ones(n, k))
    if eval_mode==True:
        dropout.eval()
    for _ in range(args.repeats):
        y = dropout(y)
    return None


def dropout_forward(eval_mode = False):
    n = args.batch_size
    k = args.n_features
    dropout = nn.Dropout()
    y = Variable(torch.ones(n, k))
    if eval_mode:
        dropout.eval()
    for _ in range(args.repeats):
        y = dropout(y)
    return None

print('FORWARD rand')
for _ in range(3):
    start = time.time()
    cdropout_forward()
    print(time_since(start), ' ContiguousDropout')
    # start = time.time()
    # dropout_forward()
    # print(time_since(start), ' Dropout')

print('FORWARD eval')
for _ in range(3):
    start = time.time()
    cdropout_forward(eval_mode=True)
    print(time_since(start), ' ContiguousDropout')
    start = time.time()
    dropout_forward(eval_mode=True)
    print(time_since(start), ' Dropout')

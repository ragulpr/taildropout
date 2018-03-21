import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from dropout import ContiguousDropout

import argparse
# python -m cProfile -s cumtime test_performance.py --repeats  10
parser = argparse.ArgumentParser(description='')
parser.add_argument('--repeats', type=int, default=10000, metavar='N')
parser.add_argument('--n-features', type=int, default=500, metavar='N')
parser.add_argument('--batch-size', type=int, default=1000, metavar='N')
parser.add_argument('--no-cuda', action='store_true', default=False,help='disables CUDA training')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
print('GPU: ',args.cuda)
print(torch.__version__)
if args.cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

n = 1000

import time, math
def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def test_ContiguousDropout():
    n = args.batch_size
    k = args.n_features
    dropout = ContiguousDropout()
    x = Variable(torch.ones(n,k))
    for _ in range(args.repeats):        
        y = dropout(x)
    return None

def test_Dropout():
    n = args.batch_size
    k = args.n_features
    dropout = nn.Dropout()
    x = Variable(torch.ones(n,k))
    for _ in range(args.repeats):        
        y = dropout(x)
    return None


start = time.time();test_ContiguousDropout();print(time_since(start),' ContiguousDropout')
start = time.time();test_Dropout();print(time_since(start),' Dropout')
start = time.time();test_ContiguousDropout();print(time_since(start),' ContiguousDropout')
start = time.time();test_Dropout();print(time_since(start),' Dropout')

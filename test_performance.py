import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from taildropout import TailDropout

import argparse
# python -m cProfile -s cumtime test_performance.py --repeats  10
parser = argparse.ArgumentParser(description='')
parser.add_argument('--repeats', type=int, default=1000000, metavar='N')
parser.add_argument('--n-features', type=int, default=1000, metavar='N')
parser.add_argument('--batch-size', type=int, default=5000, metavar='N')
parser.add_argument('--no-cuda', action='store_true',
                    default=False, help='disables CUDA training')
parser.add_argument('--time-limit', type=int,  default=None,
                    help='Maximum allowed total time in seconds')
args = parser.parse_args()

print(f'args.no_cuda: {args.no_cuda}')
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(f'GPU: { args.cuda}')
print(f'torch.__version__: {torch.__version__}')

import time
import math


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def dropout_runner(Dropout,
                   requires_grad = False,
                   eval_mode = False,
                   backward = False):
    dropout = Dropout()

    y = torch.ones(args.batch_size, args.n_features)
    if args.cuda:
        y = y.cuda()
    y = Variable(y,requires_grad=requires_grad)

    if eval_mode:
        dropout.eval()
    if requires_grad and backward:
        optimizer = torch.optim.SGD((y,),lr=0.1)

    # Work
    start = time.time()
    for _ in range(args.repeats):
        z = dropout(y)
        if requires_grad and backward:
            loss = z.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    return time_since(start),time.time() - start

total_start = time.time()

for eval_mode in [False,True]:
    for requires_grad in [True,False]:
        for backward in [True,False]:
            if backward and not requires_grad:
                break

            print('eval_mode:',eval_mode,'\t equires_grad: ',requires_grad,'\t backward: ',backward,)
            for _ in range(2):
                for Dropout in [nn.Dropout,TailDropout]:
                    timing,secs = dropout_runner(Dropout,
                                   requires_grad = requires_grad,
                                   eval_mode = eval_mode,
                                   backward = backward)
                    print(timing,'\t(',secs,'\t s total)\t',Dropout.__name__)

                    if args.time_limit is not None:
                        secs_elapsed = round(time.time() - total_start)
                        assert secs_elapsed < args.time_limit, f"Time limit exceeded: {secs_elapsed}s > {args.time_limit}s"
    
print(f'FINISH, total time: {time_since(total_start)} ({round(time.time() - total_start)}s tot)')

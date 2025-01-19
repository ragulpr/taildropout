import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from taildropout import TailDropout

import argparse

# python -m cProfile -s cumtime test_performance.py --repeats  10
parser = argparse.ArgumentParser(description='')
parser.add_argument('--repeats', type=int, default=1000000, metavar='N')
parser.add_argument('--n-features', type=int, default=512, metavar='N')
parser.add_argument('--batch-size', type=int, default=1024, metavar='N')
parser.add_argument('--no-cuda', action='store_true',
                    default=False, help='disables CUDA training')
parser.add_argument('--time-limit', type=int,  default=None,
                    help='Maximum allowed total time in seconds')
args = parser.parse_args()

print(f'args.no_cuda: {args.no_cuda}')
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(f'GPU: { args.cuda}')
print(f'torch.__version__: {torch.__version__}')
print(args)

import time
import math

import os
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['NUMEXPR_MAX_THREADS'] = '16'

# Configure PyTorch threading
torch.set_num_threads(16)  # Intra-op threads
torch.set_num_interop_threads(16)  # Inter-op threads

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def dropout_runner(dropout,
                   requires_grad = False,
                   eval_mode = False,
                   backward = False):
    device = 'cuda' if args.cuda else 'cpu'
    y = torch.ones(args.batch_size, 
                   args.n_features, 
                   requires_grad=requires_grad,
                   device = device)

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

print(f"{'Eval Mode':<12} "
      f"{'Requires Grad':<15} "
      f"{'Backward':<10} "
      f"{'Compile':<10} "
      f"{'Timing':<20} "
      f"{'Total (s)':<10} "
      f"{'Layer Type'}")
for eval_mode in [False, True]:
    for requires_grad in [True, False]:
        for backward in [True, False]:
            for compile in [False, True]:
                if backward and not requires_grad:
                    break
                    
                for _ in range(2):
                    for dropout in [TailDropout()]:  # [TailDropout(), nn.Dropout()]
                        if compile:
                            dropout.compile()

                        timing, secs = dropout_runner(
                            dropout,
                            requires_grad=requires_grad,
                            eval_mode=eval_mode,
                            backward=backward
                        )
                        
                        print(f"{str(eval_mode):<12} "
                              f"{str(requires_grad):<15} "
                              f"{str(backward):<10} "
                              f"{str(compile):<10} "
                              f"{timing:<20} {f'({secs:.2f})':<10} "
                              f"{dropout.__ne__}")

                        if args.time_limit is not None:
                            secs_elapsed = round(time.time() - total_start)
                            if secs_elapsed >= args.time_limit:
                                raise TimeoutError(
                                    f"Time limit exceeded: {secs_elapsed}s > {args.time_limit}s"
                                )

print("-" * 80)
print(f"Total time: {time_since(total_start)} ({round(time.time() - total_start)}s)")
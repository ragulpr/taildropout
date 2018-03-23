import argparse
# import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from dropout import *

criterion = nn.MSELoss()


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--n-hidden', type=int, default=100, metavar='N',
                    help='number of iterations to train (default: 10000)')
parser.add_argument('--iterations', type=int, default=100, metavar='N',
                    help='number of iterations to train (default: 10000)')
parser.add_argument('--runs', type=int, default=1, metavar='N',
                    help='repeats of experiments')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--dropout-prob', type=float, default=0.5, metavar='LR',
                    help='learning rate (default: 0.5)')
parser.add_argument('--log-interval', type=int, default=500, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

n_hidden= args.n_hidden
prob = args.dropout_prob

import pickle

def save_pickle(object,filename):
    with open(filename,'wb') as f:
            pickle.dump(object,f)

def load_pickle(filename):
    with open(filename,'rb') as f:
            return pickle.load(f)



class SequentialDropout(nn.Module):
    r""" Control using regular nn.Linear layer. Just for assert of equality.
    """
    def __init__(self):
        super(SequentialDropout, self).__init__()
        self.linear1 = nn.Linear(1,n_hidden,bias=False)
        self.linear2 = nn.Linear(n_hidden,n_hidden)
        self.linear3 = nn.Linear(n_hidden,n_hidden)
        self.linear_out = nn.Linear(n_hidden,1)
        self.dropout = ContiguousDropout(prob)
    def forward(self, x, dropout_start = None):

        x = self.dropout(self.linear1(x).tanh(),dropout_start)
        x = self.dropout(self.linear2(x).tanh(),dropout_start)
        x = self.dropout(self.linear3(x).tanh(),dropout_start)
        x = self.linear_out(x)
        return x

class RegularDropout(nn.Module):
    def __init__(self):
        super(RegularDropout, self).__init__()
        self.linear1 = nn.Linear(1,n_hidden,bias=False)
        self.linear2 = nn.Linear(n_hidden,n_hidden)
        self.linear3 = nn.Linear(n_hidden,n_hidden)
        self.linear_out = nn.Linear(n_hidden,1)
        self.dropout = nn.Dropout(prob)
        self.mask = ContiguousDropout(prob)
    def forward(self, x, dropout_start = None):
        if dropout_start is None:
            dropout_start = n_hidden
        x = self.dropout(self.mask(self.linear1(x).tanh(),dropout_start))
        x = self.dropout(self.mask(self.linear2(x).tanh(),dropout_start))
        x = self.dropout(self.mask(self.linear3(x).tanh(),dropout_start))
        x = self.linear_out(x)
        return x

class Deterministic(nn.Module):
    def __init__(self):
        super(Deterministic, self).__init__()
        self.linear1 = nn.Linear(1,n_hidden,bias=False)
        self.linear2 = nn.Linear(n_hidden,n_hidden)
        self.linear3 = nn.Linear(n_hidden,n_hidden)
        self.linear_out = nn.Linear(n_hidden,1)

        self.mask = ContiguousDropout(prob)
    def forward(self, x, dropout_start = None):
        if dropout_start is None:
            dropout_start = n_hidden
        x = self.mask(self.linear1(x).tanh(),dropout_start)
        x = self.mask(self.linear2(x).tanh(),dropout_start)
        x = self.mask(self.linear3(x).tanh(),dropout_start)
        x = self.linear_out(x)
        return x

def test_models():
    # All vals should be equal at inference mode
    vals = []
    for Model in [SequentialDropout,RegularDropout,Deterministic]:
        torch.manual_seed(1)
        model = Model()
        model.eval()
        vals.append(model(Variable(torch.randn(1000, 1))).sum().data)
        if len(vals)>2:
            assert (vals[-2]==vals[-1]).all()

test_models()

import time, math
def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def experiment():
    losses_train = []
    losses_test = []

    torch.manual_seed(seed)
    optimiser = torch.optim.Adam(model.parameters(), lr = args.lr) 
    for epoch in range(args.iterations):
        optimiser.zero_grad()        
        y = model(x+Variable(torch.randn(x.shape)*0.01))
        loss = criterion(y, y_actual)
        loss.backward()
        optimiser.step()
        if epoch%args.log_interval==0:
            losses_train.append(loss.data.cpu().numpy())
            losses_test.append(evaluate())
    return losses_test,losses_train

def evaluate(dropout_start = None):
    model.eval()
    y = model(x,dropout_start)
    model.train()
    loss = criterion(y, y_actual)
    return(loss.data.cpu().numpy())

def evaluate_conseq():
    conseq_loss = []
    for k in range(n_hidden):
        conseq_loss.append(evaluate(dropout_start = k))
    return conseq_loss

experiments_summary = []
for seed in range(args.runs):
    res = dict()
    timings = dict()
    final_eval_loss = dict()
    torch.manual_seed(seed)
    x = Variable(torch.randn(1000, 1)*2)
    y_actual = Variable(torch.randn(1000, 1))
    
    names = ['SequentialDropout','RegularDropout','Deterministic']
    models = [SequentialDropout,RegularDropout,Deterministic]
    for name,Model in zip(names,models):
        model = Model()
        start = time.time()
        losses_test,losses_train = experiment()
        print(time_since(start),'test :',losses_test[-1],'train :',losses_test[-1],name)
        timings[name] = time.time()-start
        res[name] = evaluate_conseq()
        final_eval_loss[name] = losses_test[-1]

    #     plt.semilogy(losses_test[2:],label=name)
    # plt.legend()
    # plt.show()    
    # for name in names:
    #     if name!='Deterministic':
    #         plt.semilogy(res[name],label=name)
    
    # plt.xlabel('ix used')
    # plt.ylabel('error (loss)')
    # plt.legend()
    # plt.show()
    
    experiments_summary.append([seed,res,final_eval_loss,timings])

# import pandas as pd
# df = pd.DataFrame([exp[2] for exp in experiments_summary])
# print(df.aggregate(np.mean).sort_values())
# print(df.aggregate(np.median).sort_values())

# save_pickle(df,'./results.pkl')

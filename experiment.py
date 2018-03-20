import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from dropout import *
from linearwithdropout import *

from models import *


import time, math
def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def experiment():
    criterion = nn.MSELoss()# Mean Squared Loss
    n_hidden= 50

    torch.manual_seed(seed)
    optimiser = torch.optim.Adam(model.parameters(), lr = 0.01) 
    for epoch in range(10100):
        optimiser.zero_grad()        
        y = model(x+Variable(torch.randn(1)*0.01))
        loss = criterion(y, y_actual)
        loss.backward()
        optimiser.step()
        if epoch%101==0:
            losses.append(evaluate())
    
def evaluate(dropout_start = None):
    model.eval()
    y = model(x,dropout_start)
    model.train()
    loss = criterion(y, y_actual)
    return(loss.data.numpy()[0])

def evaluate_conseq():
    conseq_loss = []
    for k in range(n_hidden):
        conseq_loss.append(evaluate(dropout_start = k))
    return conseq_loss

experiments_summary = []
for seed in range(100):
    res = dict()
    timings = dict()
    final_eval_loss = dict()
    torch.manual_seed(seed)
    x = Variable(torch.randn(1000, 1)*2)
    y_actual = Variable(torch.randn(1000, 1))
    
    names =['SequentialDropout3','SequentialDropout2','RegularDropout']
    models = [SequentialDropout3,SequentialDropout2,RegularDropout]
    for name,Model in zip(names,models):
        model = Model()
        losses = []
        start = time.time()

        experiment()
        print(time_since(start),'loss :',losses[-1],name)
        timings[name] = time.time()-start
        plt.semilogy(losses[2:],label=name)
        res[name] = evaluate_conseq()
        final_eval_loss[name] = losses[-1]
    plt.legend()
    plt.show()
    
    for name in names:
        if name!='Deterministic':
            plt.semilogy(res[name],label=name)
    
    plt.xlabel('ix used')
    plt.ylabel('error (loss)')
    plt.legend()
    plt.show()
    
    experiments_summary.append([seed,res,final_eval_loss,timings])

import pandas as pd
df = pd.DataFrame([exp[2] for exp in experiments_summary])
print(df.aggregate(np.mean).sort_values())
print(df.aggregate(np.median).sort_values())

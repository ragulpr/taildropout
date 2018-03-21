criterion = nn.MSELoss()

losses = []

n_hidden= 50
prob = 0.1

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
            dropout_start = 100000
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
            dropout_start = 100000
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
        vals.append(model(Variable(torch.randn(1000, 1))).sum().data[0])
    assert len(set(vals))==1

test_models()

import time, math
def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def experiment():
    torch.manual_seed(seed)
    optimiser = torch.optim.Adam(model.parameters(), lr = 0.001) 
    for epoch in range(1000100):
        optimiser.zero_grad()        
        y = model(x+Variable(torch.randn(x.shape)*0.01))
        loss = criterion(y, y_actual)
        loss.backward()
        optimiser.step()
        if epoch%1000==0:
            losses.append([evaluate(),loss.data.numpy()[0]])
    
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
    
    names = ['SequentialDropout']#,'RegularDropout','Deterministic']
    models = [SequentialDropout]#,RegularDropout,Deterministic]
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

# TailDropout

Check out `test.py` and `test_performance.py` to get an idea how to use it.


```
from taildropout import TailDropout
dropout = TailDropout(p=0.5,batch_dim=0, dropout_dim=1)
````
dropout is now an `nn.Module` that works just like `nn.Dropout`, applied to a tensor `x`: 

`dropout(x)` but for every `i` in batch dimension, a random `k` is drawn and all elements  `x[i,k:]` is zeroed out. Neat huh.

More description to come
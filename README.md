# TailDropout

Check out [example.ipynb](example.ipynb) or `test.py` and `test_performance.py` to get an idea how to use it.

## Choose k=1,...
![](./_figs/taildropout.gif)
## While training, randomly sample k
![](./_figs/taildropout_random.gif)
## Regular dropout chooses random subsets
![](./_figs/dropout.gif)
## Ex AutoEncoder; Sequential compression.
![](./_figs/ae.gif)

```
from taildropout import TailDropout
dropout = TailDropout(p=0.5,batch_dim=0, dropout_dim=1)
````
dropout is an `nn.Module` that works just like `nn.Dropout`, applied to a tensor `x`: 

`dropout(x)` but for every `i` in batch dimension, a random `k` is drawn and all elements  `x[i,k:]` is zeroed out. Neat huh.

More description to come

#### Citation
```
@misc{Martinsson2018,
  author = {Egil Martinsson},
  title = {TailDropout},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/naver/taildropout}},
  commit = {master}
}
```

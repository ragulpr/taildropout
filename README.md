# TailDropout ![example workflow](https://github.com/ragulpr/taildropout/actions/workflows/tests.yml/badge.svg)
**"Improving neural networks by *enforcing* co-adaptation of feature detectors"**

Check out [example.ipynb](example.ipynb) or `test.py` and `test_performance.py` to get an idea how to use it. The idea is simple. At training time, only keep a random `k` first features. Results are as expected; this makes a layer learn features that are of additive importance, just like PCA. 

I'm happy to release this since I find it to be very useful. I use this for 
* estimating the #features needed per layer
* in place of dropout for regularization
* to be able to choose a model size (after training to overfit!) that generalizes. 

## Matrix multiplication 101
At each layer, a scalar input *feature* `x[j]` of a feature vector `x` decides how far to map input into the direction `W[:,j]` of the layer output space. This is done by `W[:,j]*x[j]`:

![](./_figs/taildropout.gif)
### TailDropout: While training, randomly sample k
Teach each **k first** directions to map input to target as good as possible.
![](./_figs/taildropout_random.gif)

Each direction has decreasing probability of being used.

### Compare to regular dropout
Teach each **subset of directions** to map input to targets as good as possible.
![](./_figs/dropout.gif)

Each direction has same inclusion probability.

### Comparison to PCA
If `W` is some weights, then the SVD compression (same as PCA) is

```python
W = torch.randn(2,10)
U,s,V = torch.svd(W)
W.mm(x) == U.mm(s.diag()).mm(V.t()).mm(x) # ~True in theory
```
With `s` the eigenvalues of `W`. To use the `k` first *factors/components/eigenvectors* to represent `W`, set `s[k:]=0`. 

![](./_figs/svd.gif)

 <sub>Due to [Linear Algebra](https://en.wikipedia.org/wiki/Singular_value_decomposition), `s[2:]==0` would already be the case in example above. 
</sub>

Note that SVD compresses `W` optimally w.r.t the Euclidian norm `||W - U[:,:k] diag(s[:k]) V[:,:k]'||` for every `k`, but you want to compress each layer w.r.t the final loss function and lots of non-linearities in between!

### Example AutoEncoder; Sequential compression.
When using TailDropout on the embedding layer, `k` controlls the compression rate:

![TailDropout](./_figs/ae-taildropout.gif)

Here even with `k=1` the resulting 1d-scalar embedding apparently separates shoes and shirts. 

Compare this to how regular dropout works. Well, it's quite more random.
![Regular dropout](./_figs/ae-dropout.gif)


## Usage
TailDropout is an `nn.Module` that works just like `nn.Dropout`, applied to a tensor `x`: 
```python
from taildropout import TailDropout
dropout = TailDropout(p=0.5,batch_dim=0, dropout_dim=-1)
y = dropout(x)
```
See [example.ipynb](example.ipynb) for complete examples.

To use it for pruning or estimating the optimal size of hidden dim, calculate n_features vs loss and create a [scree plot](https://en.wikipedia.org/wiki/Scree_plot#:~:text=In%20multivariate%20statistics%2C%20a%20scree,principal%20component%20analysis%20(PCA).)
```python
for k in range(n_features):
  model.dropout.set_k(k)
  y_pred = model(x) # calculate loss using only k features
  loss = criterion(y, y_pred)

plt.plot(range(n_features), loss)
```

#### Pseudocode
```python
# x = input from previous layer
# L = parameter controlling dropout-probability
for i in range(n_batch):
    k = ~Exponential(L)
    x[i,k:] = 0 
```
Note, the actual implementation is **much** faster, vectorized and made to be pytorch 0.2x, 0.3x, 0.4x, 1x GPU compatible. Tested for Pytorch <0.4.1 it yielded a significant speedup over regular dropout.

## Details
#### Training vs Inference
```python
dropout = TailDropout()
dropout(x) # random
dropout.eval() 
dropout(x) # Identity function
dropout.set_k(k)
dropout(x) # use first k features 
```

#### Sequences
"Recurrent dropout" == Keep mask constant over time. Popular approach.
```python
x = torch.randn(n_timesteps,n_sequences,n_features)

gru = nn.GRU(n_features,n_features)
taildropout = TailDropout(batch_dim = 1, dropout_dim = 2)

x, _ = gru(x)
x = taildropout(x)
```
If you want to have mask vary for each timestep and sequence
```python
taildropout = TailDropout(batch_dim = [0,1], dropout_dim = 2)
```

#### Images
"2d Dropout" == Keep mask constant over spatial dimension. Popular approach.
```python
x = torch.randn(n_batch,n_features,n_pixels_x,n_pixels_y)

cnn = nn.Conv2d(n_features,n_features, kernel_size)
taildropout = TailDropout(batch_dim = 0, dropout_dim = 1)

x = cnn(x)
x = taildropout(x)
```

#### BatchNorm
Same as with regular dropout; batchnorm *before* dropout.
```python
layer = nn.Sequential(
    nn.Linear(n_features,n_features),
    nn.BatchNorm1d(n_features),
    nn.ReLU(),
    TailDropout()
    )
```

##### Compression/regularization ratio is very large!
If you don't care much about regularization, dropout probability in order 1e-5 still 
seems to give good compression effect. I typically use `TailDropout(p=0.001)` to get both. 

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

#### Aknowledgments
This work was open sourced 2025 but work primarily done in 2018 at [Naver Clova/Clair](https://research.clova.ai/). Big thanks to [Minjoon Seo](https://seominjoon.github.io/) for the original inspiration from his work on [Skim-RNN](https://arxiv.org/abs/1711.02085) and [Ji-Hoon Kim](https://scholar.google.co.kr/citations?user=1KdhN5QAAAAJ&hl=ko) [Adrian Kim](https://scholar.google.co.kr/citations?user=l6lDgpgAAAAJ&hl=ko), [Jaesung Huh
](https://scholar.google.com/citations?user=VDMZ-pQAAAAJ&hl=en), [Prof. Jung-Woo Ha](https://scholar.google.com/citations?user=eGj3ay4AAAAJ&hl=en) and [Prof. Sung Kim](https://scholar.google.com/citations?user=JE_m2UgAAAAJ&hl=en) for valuable discussions and feedback.

I'm sure this simple idea has been implemented before 2018 (which I was unaware of at the time) or after (which I have not had time to look for). Please let me know if there's anything relevant I should cite.

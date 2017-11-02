# 何品萱 (106062553)
## Homework3-Policy-Gradient report

TA: try to elaborate the algorithms that you implemented and any details worth mentioned.
#### Algorithms and code details


##### Problem 1: construct a neural network to represent policy
Use TensorFlow to construct a 2-layer neural network as stochastic policy.

##### Problem 2: compute the surrogate loss
accumulated discounted rewards
<p align="center"><img src="imgs/discounted rewards.jpg" width=30% /></p>

surrogate loss
<p align="center"><img src="imgs/surrogate loss.jpg" width=50% /></p>

```
  surr_loss = -tf.reduce_mean(log_prob * self._advantages)
```

##### Problem 3: reduce the variance of our gradient estimate
Change the loss term into:
<p align="center"><img src="imgs/reduce the variance.jpg" width=50% /></p>
<p align="center"><img src="imgs/V.jpg" width=20% /></p>

```
  a = r - b
```



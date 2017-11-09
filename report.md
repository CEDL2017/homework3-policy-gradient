# Homework3-Policy-Gradient report

TA: try to elaborate the algorithms that you implemented and any details worth mentioned.

## Problem 1: Construct a neural network to represent policy

```python
# fc1
layer1 = tf.layers.dense(
    inputs=self._observations,
    units=hidden_dim,
    activation=tf.nn.tanh,  # tanh activation
    kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
    bias_initializer=tf.constant_initializer(0.1),
    name='fc1'
)
# fc2
probs = tf.layers.dense(
    inputs=layer1,
    units=out_dim,
    activation=tf.nn.softmax,
    kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
    bias_initializer=tf.constant_initializer(0.1),
    name='fc2'
)        
```

> 建立兩個Fully-connected layer 的隱藏層<br>
> 參考視頻 莫煩<br>
> https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/5-1-policy-gradient-softmax1/
<br>
<br> 

## Problem 2: compute the surrogate loss
```python
surr_loss = -tf.reduce_mean(log_prob * self._advantages)
```
<p align="center"><img src="https://morvanzhou.github.io/static/results/reinforcement-learning/5-1-1.png" height="300"/></p>

# 105061516 Homework3-Policy-Gradient report

TA: try to elaborate the algorithms that you implemented and any details worth mentioned.


## Problem 1: Construct a neural network to represent policy

```python
self.fc1 = tf.layers.dense(inputs=self._observations, units=hidden_dim, activation=tf.nn.tanh, name="fc1")
probs = tf.layers.dense(inputs=self.fc1, units=out_dim, activation=tf.nn.softmax, name="fc2")
```

> 用Fully-connected layer 建立一層隱藏層<br>
> 共 2 個 layers，上方為輸入層至隱藏層的連結，下方為隱藏層至輸出層的連結。
<br>
<br> 

## Problem 2: compute the surrogate loss

```python
loss2rew = log_prob * self._advantages
surr_loss = -tf.reduce_mean(loss2rew)
```
> 將loss乘以一個權重<br>
> 再將其轉為reward。
<br>
<br>

##　Problem 3: Compare with baseline

```python
a = r - b
```

> 將訓練中得到的reward與baseline相比，得出其差異。

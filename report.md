# 105061525 許菀庭
# Homework3-Policy-Gradient report

## Problem 1: Construct a Neural Network to Represent Policy

```python
hidden1 = tf.layers.dense(inputs=self._observations, units=hidden_dim, activation=tf.nn.tanh)
probs = tf.layers.dense(inputs=hidden1, units=out_dim, activation=tf.nn.softmax)
```
我用tensorflow中的tf.layers.dense來實現一個2-layer neural network。tf.layers.dense代表的就是fully-connected layer，輸入inputs、output dimension 和 activation function 就可以。這個network代表的是policy，因此inputs是observations，outputs是對應此observation，各個action的probabilities。

## Problem 2: Compute the Surrogate Loss

```python
surr_loss = -tf.reduce_mean(self._advantages * log_prob)
```
這裡實現了policy gradient的surrogate loss，計算每個episode的每個time step的 advantage * log pi(a|s)，然後取平均。因為advantage是越大越好，但optimizer是要minimize loss，所以這裡加一個負號。


## Problem 3: Reduce Variance Using a Baseline

```python
a = r - b
# a is advantages
# r is rewards
# b is baseline values
```
這個步驟把reward減掉baseline，baseline是用一個estimate的value function算出來的。這個的目的是要讓我們的model在採取一個action的reward會比baseline還高的時候，才去encourage 這個action。這麼做可以reduce variance，讓training過程更穩定。


## Problem 4: Compare the Results Before and After Adding Baseline

### why the baseline won't introduce bias

加上baseline並不會改變原本的expectation: 


## Problem 5: Actor-Critic Algorithm (With Bootstrapping)




## Problem 6: Generalized Advantage Estimation



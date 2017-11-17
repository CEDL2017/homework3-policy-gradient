# Homework3-Policy-Gradient report
## Problem 1 : Construct a neural network to represent policy


```
  fc1 = tf.contrib.layers.fully_connected(self._observations, hidden_dim, activation_fn=tf.nn.tanh)
  fc2 = tf.contrib.layers.fully_connected(fc1, out_dim, activation_fn=None)
  probs = tf.contrib.layers.softmax(fc2)
```
用兩層 fully-connected 建立 policy network。定義fully-connected layer 需要給定input和 output neuron number, 使用tanh當作 activation function。

## Problem 2 : Compute the Surrogate Loss
```
surr_loss = -tf.reduce_mean(self._advantages * log_prob)
```
根據公式，surrogate loss是對應的advantage乘上機率，取平均。因為tensorflow只能做minimize的optimization，所以要把reward取負號得到surrogate loss。

## Problem 3 : Reduce Variance Using a Baseline

```
 # YOUR CODE HERE >>>>>>
 a = r - b
 # <<<<<<<<

p["returns"] = r
p["baselines"] = b
p["advantages"] = (a - a.mean()) / (a.std() + 1e-8) # normalize
```
將 reward 減掉baseline。Baseline是estimate的value function，減掉baseline讓reward必須大於目前的value，才鼓勵 model 做這樣的動作。再數學上來看是減少了variance可以讓training更穩定。

## Problem 4 : Compare the Results Before and After Adding Baseline

### 1. Why the baseline won't introduce bias

這是 surrogate loss 的公式

在reward 減掉 baseline，多了這一項

baseline只要跟action無關，微分後都是0，完全不影響結果。


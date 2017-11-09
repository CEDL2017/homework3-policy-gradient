# Homework3-Policy-Gradient report
student name/ID: 翁正欣/106062577
## Problem 1: construct a neural network to represent policy
```
with tf.variable_scope("fc1"):
      weights = tf.Variable(tf.truncated_normal(shape=[in_dim, hidden_dim], seed=0))
      biases = tf.Variable(tf.truncated_normal(shape=[hidden_dim], seed=0))
      logit = tf.nn.xw_plus_b(self._observations, weights, biases)
      act = tf.tanh(logit)

  with tf.variable_scope("fc2"):
      weights = tf.Variable(tf.truncated_normal(shape=[hidden_dim, out_dim], seed=0))
      biases = tf.Variable(tf.truncated_normal(shape=[out_dim], seed=0))
      logit = tf.nn.xw_plus_b(act, weights, biases)
      softmax = tf.nn.softmax(logit)
  probs = softmax
```
## Problem 2: compute the surrogate loss
```
surr_loss = -tf.reduce_mean(log_prob*self._advantages)
```
## Problem 3: Use baseline to reduce the variance of our gradient estimate
in this problem, I substract **baseline** from **returns**.
```
a = r-b
```
## Problem 4: train without baseline
in this problem, I did not use baseline and use returns as advantage.
## Problem 5: Actor-Critic algorithm (with bootstrapping)
in this problem, I implement Actor-Critic with bootstrapping.
```
def discount_bootstrap(x, discount_rate, b):
    b = np.concatenate([b,[0]])
    y = x + discount_rate*b[1:]
    return y
```
## Problem 6: Generalized Advantage Estimation
in this problem, I discount advantage by GAE.
```
"""
y[0] = x[0] + discount(x[1],1) + discount(x[2],2) + ... + discount(x[len(x)-1], len(x)-1)
y[1] = x[1] + discount(x[2],1) + discount(x[3],2) + ... + discount(x[len(x)-1], len(x)-2)
...
y[n] = x[n] + discount(x[n], 1) + ... + discount(x[len(x)-1], len(x)-n+1)
     = x[n] + discount(y[n+1],1)
"""
a = util.discount(a, discount_rate*LAMBDA)
```
## Experiments
all experiment are under the same random seed for environment and network initial weights.

|problem3|problem4|problem5|problem6|
|---|---|---|---|
|![](https://i.imgur.com/F1hzpO2.png)|![](https://i.imgur.com/DDDN8qn.png)|![](https://i.imgur.com/Qw1bNwv.png)|![](https://i.imgur.com/TcYEveS.png)|
|![](https://i.imgur.com/CIgBDGx.png)|![](https://i.imgur.com/aJuHqco.png)|![](https://i.imgur.com/SY5MnRO.png)|![](https://i.imgur.com/v02OZ37.png)||

## Conclusion
setting for problem4 converges the fastest in this task. maybe I implement something wrong...

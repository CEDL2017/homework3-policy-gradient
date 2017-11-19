# CEDL2017 HW3 Report

## Problem 1
In problem 1, I use `tf.contrib.layers` to construct my 2-layer neural network, nothing special here.
```python
h1 = tf.contrib.layers.fully_connected(self._observations, num_outputs=hidden_dim, activation_fn=tf.tanh)
h2 = tf.contrib.layers.fully_connected(h1, num_outputs=out_dim, activation_fn=None)
probs = tf.nn.softmax(h2)
```

## Problem 2
Here we calculate the surrogate loss:
```python
# 1. self._advantages is a vector containing Ri for all timestep.
# 2. log_prob is a vector containing log(pi(a|s)) for all timestep.  
# 3. Element-wise product of Ri and log_prob.
# 4. Sum over time, divide by T to get surrogate loss.
neg_log_prob = -log_prob # Minimizing negative log-likelihood is same as maximizing log-likelihood.
surr_loss = tf.reduce_mean(neg_log_prob*self._advantages)
```
Note that the optimizer in TensorFlow minimizes loss, thus we simply maximizes the log-likelihood by adding negative sign, which is equivalent.

## Problem 3
In problem 3, we need to modify the reward to **advantage**, which is just reward subtracted by baseline:
```python
a = r - b
```

| Loss | Average Return | Average Advantage Variance |
| ---- | ------ | -------- |
|![](https://i.imgur.com/CI5SzP1.png) | ![](https://i.imgur.com/ecYw6FZ.png) | ![](https://i.imgur.com/3K5Hlp8.png)

## Problem 4
We found that after removing the baseline, the average advantage variance explodes, but the average return still can exceed 195 within about 80 iterations. The reason why the baseline won't introduce the bias can be proved as follows:

| Loss | Average Return | Average Advantage Variance |
| ---- | ------ | -------- |
| ![](https://i.imgur.com/oRClDgo.png) |![](https://i.imgur.com/TVwJWpp.png) |![](https://i.imgur.com/ZVIds7Y.png)

## Problem 5
In problem 5, we modify the advantage from the original version to actor-critic algorithm whose advantage function is one-step bootstrap. But the return cannot achieve to 195 within 200 iterations, may need more iterations to learn. Moreover, as shown in the following figure, the loss is very unstable.

| Loss | Average Return | Average Advantage Variance |
| ---- | ------ | -------- |
| ![](https://i.imgur.com/BYYoKv9.png) | ![](https://i.imgur.com/oZ2AaGT.png) | ![](https://i.imgur.com/yRGRWIY.png)

## Problem 6
In problem 6, we further modify the one-step bootstrap to the form of Generative Advantage Estimation (GAE). After this modification, the agent can achieve to 195 within about 80 iterations, but the average advantage variance is higher than the one in problem 3.

| Loss | Average Return | Average Advantage Variance |
| ---- | ------ | -------- |
| ![](https://i.imgur.com/rFjgB4E.png) | ![](https://i.imgur.com/O0FqYC9.png) | ![](https://i.imgur.com/uFMMSvU.png)

# Homework3-Policy-Gradient report

## Problem 1: Construct a neural network to represent policy

In this homework, we use simple 2-layer neural network to construct policy.

![](images/nn.jpg)

```python
hidden = tf.layers.dense(self._observations, hidden_dim, activation=tf.nn.tanh)
output = tf.layers.dense(hidden, out_dim)
probs = tf.nn.softmax(output)
```

## Problem 2: Compute the surrogate loss

![](images/surrogate_loss_1.png)

![](images/surrogate_loss_2.png)

```python
surr_loss = tf.reduce_mean(tf.multiply(log_prob, self._advantages))
surr_loss = tf.multiply(surr_loss, -1)
```

## Problem 3: Baseline Method

Use baseline to reduce the variance of our gradient estimate.

```python
a = r - b
```

## Problem 4: Without Baseline

Why baseline won't introduced bias?

![](images/baseline_bias.png)

##### Compare with and without baseline

<table>
    <tr>
        <td> With Baseline </td>
        <td> <img src="images/with_baseline_loss.png"/> </td>
        <td> <img src="images/with_baseline_avg_return.png"/> </td>
    </tr>
    <tr>
        <td> Without Baseline </td>
        <td> <img src="images/without_baseline_loss.png"/> </td>
        <td> <img src="images/without_baseline_avg_return.png"/> </td>
    </tr>
</table>

> With baseline, we can reduce the variance of policy, so that training can be more stable than without baseline one.

## Problem 5: Actor-Critic algorithm with bootstrapping

<table>
    <tr>
        <td> <img src="images/actor_critic_loss.png"/> </td>
        <td> <img src="images/actor_critic_avg_return.png"/> </td>
    </tr>
</table>

```python
y = x + discount_rate * np.append(b, 0)[1:]
return y
```

## Problem 6: Generalized Advantage Estimation

<table>
    <tr>
        <td> <img src="images/gae_loss.png"/> </td>
        <td> <img src="images/gae_avg_return.png"/> </td>
    </tr>
</table>

```python
a = util.discount(a, self.discount_rate * LAMBDA)
```

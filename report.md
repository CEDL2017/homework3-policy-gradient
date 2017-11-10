# Homework3-Policy-Gradient report

## Problem 1

The two fc-layers were constructed as follows:

```
hidden_layer = tf.contrib.layers.fully_connected(self._observations, hidden_dim, activation_fn=tf.tanh)
```
```
probs = tf.contrib.layers.fully_connected(hidden_layer, out_dim, activation_fn=tf.nn.softmax)
```

Note the little trick that the output layer was activated with `tf.nn.softmax`.

## Problem 2

Since the optimizer will minimize the given loss function, an negative symbol should be added to the surrogate loss ![surr_loss](https://latex.codecogs.com/gif.latex?\frac{1}{T}\sum_{t=0}^T&space;log\pi_\theta(a_t^i&space;|&space;s_t^i)&space;*R_t^i).

Thus, the code should be
```
surr_loss = -tf.reduce_mean(log_prob * self._advantages)
```

## Problem 3

In problem 3, a baseline predicted value was subtracted from the reward at each timestep to reduce the variance of our gradient estimate. It's could be done with a simple line code:

```
a = r - b
``` 

Variable `a` would be assigned to `data["advantages"]` after that, which was used to compute the surrogate loss.

![](img/prob3_loss.png)

This problem could be solved at 58 iterations.

![](img/prob3_reward.png)

## Problem 4

For problem 4, the vanilla policy gradient, just simply set the variable `baseline` to `None`.

![](img/prob4_loss.png)

It took a little more iterations (68 iterations) to solve.

![](img/prob4_reward.png)

## Problem 5

In problem 5 the advantage function was revised to ![](https://latex.codecogs.com/gif.latex?A_t^i&space;=&space;r_t^i&space;&plus;&space;\gamma*V_{t&plus;1}^i&space;-&space;V_t^i), which was implemented as ![](https://latex.codecogs.com/gif.latex?A^i&space;=&space;discount\_bootstrap(r,&space;\gamma,&space;V^i)&space;-&space;V^i).

For the purpose of readability, a for-loop was used to produce the output numpy array.

```
y = np.zeros(x.shape)
for t in range(len(x)-2, -1, -1):
    y[t] = x[t] + discount_rate * b[t+1]
return y
```

![](img/prob5_loss.png)

This algorithm seems to be unstable, but it can still solve the problem after several attempts.

![](img/prob5_reward.png)

## Problem 6

The bootstrapping in **Problem 5** was multiplied by a coefficient ![](https://latex.codecogs.com/gif.latex?(\gamma\lambda)^l) here.

We may just call the provided function.
```
a = util.discount(a, self.discount_rate * LAMBDA)
```

![](img/prob6_loss.png)

It's slightly faster to solve the problem than Actor-Critic (with bootstrapping)

![](img/prob6_reward.png)

## Supplementary Material

I also have implemented the Monte-Carlo Policy Gradient in PyTorch.

Take a look [[here]](https://github.com/sonic1sonic/Monte-Carlo-Policy-Gradient-REINFORCE).
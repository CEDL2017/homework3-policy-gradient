# Homework3-Policy-Gradient report
## Problem 1
Following instructions: Use `tanh` as the activation function of the first hidden layer, and append softmax layer after the output of the neural network to get the probability of each possible action.
```python
layer1 = tf.contrib.layers.fully_connected(self._observations, hidden_dim, activation_fn=tf.tanh)
layer2 = tf.contrib.layers.fully_connected(layer1, out_dim, activation_fn=tf.nn.softmax)
probs = layer2
```
## Problem 2
Compute surrogate loss and assign it to variable `surr_loss`. Since we need to maximize it not minimize it, so we turn it into negative.
```python
surr_loss = tf.reduce_mean(-log_prob * self._advantages)
```

## Problem 3
Simply subtract the variance and then assign the result to the variable `a`.
```python
 a = r - b
```
![](https://i.imgur.com/78ARcKp.png)
![](https://i.imgur.com/TfxMgpZ.png)

## Problem 4
Curve without baseline substraction.

![](https://i.imgur.com/hAIowgF.png)
![](https://i.imgur.com/jxAnLjv.png)

## Problem 5
Following `y = r(s_t,a,s_{t+1}) + \gamma*V_t`
```python
b = np.append(b[1:], 0)
y = x + discount_rate * b
```

![](https://i.imgur.com/fonXELu.png)
![](https://i.imgur.com/20uIuUT.png)


## Problem 6
Generalized Advantage Estimation
```python
a = util.discount(a, self.discount_rate * LAMBDA)
```

![](https://i.imgur.com/HnAgfXD.png)
![](https://i.imgur.com/8vN4nNC.png)




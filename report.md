# Homework3-Policy-Gradient report

## Problem 1 

Construct the network as the instruction says.

```python
        fc1 = tf.contrib.layers.fully_connected(
        		inputs = self._observations,
        		num_outputs = hidden_dim,
        		activation_fn = tf.nn.tanh
        	)
        fc2 = tf.contrib.layers.fully_connected(
                inputs = fc1,
                num_outputs = out_dim,
                activation_fn = None
            )
       	probs = tf.nn.softmax(fc2)
```

## Problem 2

Tensorflow's optimizer is set to find the minimum of the loss function.
However, we need to maximize the loss in this case.
Therefore, we compute the surrogate loss as the formula directs and make it negative.

```python
        surr_loss = -tf.reduce_mean(tf.multiply(log_prob, self._advantages))
```

## Problem 3

The insturction is to use the baseline prediction to minimize the variation.
All we need to do is substract the baseline prediction value from the discounted reward.
The prediction and reward are all calculated beforehand in the code.

```python
            a = r - b
```

Then we have the loss curve and advantage curve like below:
![](https://i.imgur.com/y8McOzN.png)


## Problem 4

Here's the curve without baseline substraction.
![](https://i.imgur.com/wDSGwEf.png)
The variance chages rapidly.
The reason why the baseline won't introduce bias, we may refer to policy gradient material in CS294.
The slide offers a proof that the b(bias) in the formula will be 0 after the substraction.
![](https://i.imgur.com/IUTcMnE.png)

## Problem 5

The following code suffices the bootstrap requirement:

```python
    b = np.append(b[1:], 0)
    return x + discount_rate * b
```

And the result:
![](https://i.imgur.com/9iMeFxz.png)

## Problem 6

In porlbem 6, we implement GAE as the formula indicates. Here's the code:

```python
            a = util.discount(a, self.discount_rate * LAMBDA)
```

Result:

![](https://i.imgur.com/u3APTpD.png)

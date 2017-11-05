# 何品萱 (106062553)
## Homework3-Policy-Gradient report


#### Algorithms and code details


##### Problem 1: construct a neural network to represent policy
Use TensorFlow to construct a 2-layer neural network as stochastic policy.

```
  fc1 = tf.layers.dense(inputs=self._observations , units=hidden_dim, activation=tf.nn.tanh)
  fc2 = tf.layers.dense(inputs=fc1 , units=out_dim, activation=tf.nn.softmax) 
  probs = fc2
```

##### Problem 2: compute the surrogate loss
accumulated discounted rewards
<p><img src="imgs/discounted rewards.jpg" width=20% /></p>

surrogate loss
<p><img src="imgs/surrogate loss.jpg" width=40% /></p>

```
  surr_loss = -tf.reduce_mean(log_prob * self._advantages)
```

##### Problem 3: reduce the variance of our gradient estimate
Change the loss term into:
<p><img src="imgs/reduce the variance.jpg" width=40% /></p>
<p><img src="imgs/V.jpg" width=20% /></p>

```
  a = r - b
```

###### Results


##### Problem 4
Compare the variance and performance before and after adding baseline.

with the baseline
<p align="center"><img src="imgs/.jpg" width=50% /></p>
smaller variance 

be biased

without the baseline
<p align="center"><img src="imgs/.jpg" width=50% /></p>
larger variance

unbiased

Why the baseline won't introduce bias?



##### Problem 5: Actor-Critic algorithm (with bootstrapping)
use the one-step bootstrap for the advantage function
<p><img src="imgs/bootstrapping.jpg" width=20% /></p>

```
    b = np.roll(b, -1)    #Vt+1
    b[-1] = 0
    y = x + discount_rate *b
```

##### Problem 6: Generalized Advantage Estimation
use "Generalized Advantage Estimation" for the advantage function

compromise the above two estimation methods(REINFORCE and TD)
<p><img src="imgs/GAE.jpg" width=20% /></p>
<p><img src="imgs/s.jpg" width=20% /></p>

```
  a = util.discount(a, self.discount_rate * LAMBDA)
```

###### Results



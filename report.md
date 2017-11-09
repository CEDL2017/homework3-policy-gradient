# Homework3-Policy-Gradient report
### 105062576 陳則銘
  
## Introduction
In this homework, we will use a neural network to learn a parameterize policy that can select action without consulting a value function. A value function may still be used to learn the policy weights, but is not required for action selection.
  
There are some advantage of the policy-based algorithms:
* Policy-based methods also offer useful ways of dealing with continuous action spaces
* For some tasks, the policy function is simpler and thus easier to approximate.
  
We will use CartPole-v0 as environment in this homework.
  
## Environment
* Python 3.6.3
* OpenAI gym
* numpy
* matplotlib
* ipython
* tensorflow
* scipy
* licecap

## Implementation
Policy gradient method is a family of RL algorithms that parameterizes the policy directly. The simplest advantage that policy parameterization may have over action-value parameterization is that the policy may be a simpler function to approximate.

### Problem 1: construct a neural network to represent policy
Use TensorFlow to construct a 2-layer neural network as stochastic policy. The hidden layer should be fully-connected. Use tanh as the activation function of the first hidden layer, and append softmax layer after the output of the neural network to get the probability of each possible action.
  
```
fc1 = tf.contrib.layers.fully_connected( inputs = self._observations, 
                                         num_outputs = hidden_dim, 
                                         activation_fn = tf.nn.tanh )
fc2 = tf.contrib.layers.fully_connected( inputs = fc1, 
                                         num_outputs = out_dim, 
                                         activation_fn = tf.nn.softmax )
probs = fc2
```
  
### Problem 2: compute the surrogate loss
Tensorflow has a function to minimize loss, but we should add '-' due to the minimization, while we want maximization here.
  
```
surr_loss = tf.reduce_mean(-log_prob*self._advantages)
```
  
### Problem 3: Sampling-based Tabular Q-Learning
Use baseline to reduce the variance of our gradient estimate. The advantage estimate A = R - b.
In order to reduce the variance of the gradient estimator, a constant baseline can be subtracted from the gradient.
```
a = r - b
```
Because the initial weight is randomly choose every time, so sometimes iterations may sometimes exceed 80.
  
results:
  
![](/policy_gradient/pb3_1.png)
![](/policy_gradient/pb3_2.png)  
  
### Problem 4: Compare baseline with non-baseline
with baseline:
  
![](/policy_gradient/pb4_3.png)
![](/policy_gradient/pb4_4.png)  
  
Non-baseline:
  
![](/policy_gradient/pb4_1.png)
![](/policy_gradient/pb4_2.png)

The difference from using baseline and non-baseline is the change of variance. We can show that from images above. So, by using baseline, we can reduce the variance of our gradient estimate without changing expectation. 

### Problem 5: Actor-Critic algorithm (with bootstrapping)
Actor is based on policy-iteration methods such as Policy Gradient, and Critic is based on value-iteration methods such as Q-learning. Actor-Critic combines the benefits of both approaches that actor improves the current policy, and critic evaluates the current policy.

We use the one-step bootstrap for the advantage function, which change the <a href="https://www.codecogs.com/eqnedit.php?latex=A^i_t&space;=&space;(R^i_t-V^i_t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?A^i_t&space;=&space;(R^i_t-V^i_t)" title="A^i_t = (R^i_t-V^i_t)" /></a> in problem 3 into:
<a href="https://www.codecogs.com/eqnedit.php?latex=A^i_t&space;=&space;r^i_t&space;&plus;&space;\gamma&space;*&space;V^i_{t&plus;1}-V^i_t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?A^i_t&space;=&space;r^i_t&space;&plus;&space;\gamma&space;*&space;V^i_{t&plus;1}-V^i_t" title="A^i_t = r^i_t + \gamma * V^i_{t+1}-V^i_t" /></a>
```
def discount_bootstrap(x, discount_rate, b):
    b = np.append(b[1:],0)
    y = x + discount_rate*b
    return y
```
  
```
r = util.discount_bootstrap(p["rewards"], self.discount_rate, b)
a = r - b
```

result:
  
![](/policy_gradient/pb5_1.png)
![](/policy_gradient/pb5_2.png)

The result doesn't look well. The iteration numbers exceed 200, and the performance of average return is unstable.

### Problem 6:
Here, we use a novel advantage function called "Generalized Advantage Estimation", which introduces one hyperparameter λ to compromise the above two estimation methods.
Assume the <a href="https://www.codecogs.com/eqnedit.php?latex=\delta^i_t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\delta^i_t" title="\delta^i_t" /></a> represent the i-step bootstrapping. 
The generalized advantage estimation will be:
<a href="https://www.codecogs.com/eqnedit.php?latex=A^{GAE}_t=\sum_{t=0}^{\inf}&space;(\gamma&space;\lambda)^l&space;\delta_{t&plus;1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?A^{GAE}_t=\sum_{t=0}^{\inf}&space;(\gamma&space;\lambda)^l&space;\delta_{t&plus;1}" title="A^{GAE}_t=\sum_{t=0}^{\inf} (\gamma \lambda)^l \delta_{t+1}" /></a>

```
a = util.discount(a, self.discount_rate*LAMBDA)
```

result:  
  
LAMBDA = 0.98
  
![](/policy_gradient/pb6_1.png)
![](/policy_gradient/pb6_2.png)
  
LAMBDA = 0.58

![](/policy_gradient/pb6_3.png)
![](/policy_gradient/pb6_4.png)

 







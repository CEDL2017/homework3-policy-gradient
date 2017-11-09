# Homework3-Policy-Gradient report
陳冠元(105065530)

## Problem 1: construct a neural network to represent policy
In this part, I use tf.layers.dense() function to build neural networks for solving the problem. Details as follow:
```
        # YOUR CODE HERE >>>>>>
        l1 = tf.layers.dense(self._observations, hidden_dim, tf.nn.tanh)
        l2 = tf.layers.dense(l1, out_dim)
        probs = tf.nn.softmax(l2)
        # <<<<<<<<
```

## Problem 2: compute the surrogate loss   
In this part, I added 1 line to solve the problem via the given hints.
```
        # YOUR CODE HERE >>>>>>
        surr_loss = tf.reduce_mean((-log_prob)*self._advantages)
        # <<<<<<<<
```

<div align=left>
<img src="https://github.com/guan-yuan/homework3-policy-gradient/blob/master/outputs/p1.PNG" width = "100%" alt=""/>
</div>
    
[Reference](https://github.com/CEDL2017/homework3-policy-gradient/blob/master/Lab3-policy-gradient.ipynb)


## Problem 3
In this part, I added 1 line to solve the problem via the given hints.
```
         # YOUR CODE HERE >>>>>>
         a = r - b
         # <<<<<<<<
```

<div align=left>
<img src="https://github.com/guan-yuan/homework3-policy-gradient/blob/master/outputs/3.png" width = "100%" alt=""/>
</div>

#### Results:

<div align=left>
<img src="https://github.com/guan-yuan/homework3-policy-gradient/blob/master/outputs/4.png" width = "100%" alt=""/>
</div>


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
<div align=left>
<img src="https://github.com/guan-yuan/homework3-policy-gradient/blob/master/outputs/p1.PNG" width = "100%" alt=""/>
</div>

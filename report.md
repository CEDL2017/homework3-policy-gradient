# Homework3-Policy-Gradient report

TA: try to elaborate the algorithms that you implemented and any details worth mentioned.
### Problem 1
build a two layer NN
+ Code
```python
hidden_layer1 = tf.contrib.layers.fully_connected(self._observations, hidden_dim, activation_fn=tf.tanh) 
hidden_layer2 = tf.contrib.layers.fully_connected(hidden_layer1, out_dim, activation_fn=None) 
probs = tf.nn.softmax(hidden_layer2) 
```
   
### Problem 2
+ Code
```python
surr_loss = -tf.reduce_mean(tf.multiply(log_prob, self._advantages))
```
### Problem 3
Because we have baseline b and reward r, all we need to do is a = r - b.
+ Code
```python
a = r - b
```
![](https://i.imgur.com/iVfUkwI.png) ![](https://i.imgur.com/z9AnuHS.png)
Finish at 78 iters, However 4/5 of my training exceed 80 iters, even some of them exceed 100 iters.

### Problem 4
+ Code
```python
baseline = None
```
![](https://i.imgur.com/6EESZ5u.png) ![](https://i.imgur.com/TGzCt8K.png)

Removing the baseline would'n introduce bias.

### Problem 5
+ Code
```python
b_next = np.append(b[1:], 0)
y = x + discount_rate * b_next
```
![](https://i.imgur.com/jXQSQGi.png)
1[](https://i.imgur.com/javH0dL.png)

### Problem 6
+ Code
```python
a = util.discount(a, self.discount_rate * LAMBDA)
```
![](https://i.imgur.com/yzH4rfr.png)
![](https://i.imgur.com/2F9VqJz.png)


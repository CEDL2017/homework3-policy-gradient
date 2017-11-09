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
Iteration 1: Average Return = 14.43
Iteration 2: Average Return = 15.61
Iteration 3: Average Return = 15.37
Iteration 4: Average Return = 17.1
Iteration 5: Average Return = 18.48
Iteration 6: Average Return = 18.45
Iteration 7: Average Return = 18.04
Iteration 8: Average Return = 21.42
Iteration 9: Average Return = 23.96
Iteration 10: Average Return = 26.73
Iteration 11: Average Return = 29.31
Iteration 12: Average Return = 31.95
Iteration 13: Average Return = 31.82
Iteration 14: Average Return = 35.65
Iteration 15: Average Return = 39.26
Iteration 16: Average Return = 40.12
Iteration 17: Average Return = 44.32
Iteration 18: Average Return = 45.32
Iteration 19: Average Return = 45.02
Iteration 20: Average Return = 47.66
Iteration 21: Average Return = 51.76
Iteration 22: Average Return = 51.98
Iteration 23: Average Return = 55.43
Iteration 24: Average Return = 54.57
Iteration 25: Average Return = 57.17
Iteration 26: Average Return = 59.95
Iteration 27: Average Return = 59.81
Iteration 28: Average Return = 60.25
Iteration 29: Average Return = 65.64
Iteration 30: Average Return = 66.94
Iteration 31: Average Return = 72.28
Iteration 32: Average Return = 64.43
Iteration 33: Average Return = 72.87
Iteration 34: Average Return = 69.77
Iteration 35: Average Return = 76.65
Iteration 36: Average Return = 73.67
Iteration 37: Average Return = 76.77
Iteration 38: Average Return = 79.44
Iteration 39: Average Return = 76.08
Iteration 40: Average Return = 86.19
Iteration 41: Average Return = 82.61
Iteration 42: Average Return = 81.65
Iteration 43: Average Return = 87.79
Iteration 44: Average Return = 89.81
Iteration 45: Average Return = 97.28
Iteration 46: Average Return = 102.56
Iteration 47: Average Return = 100.18
Iteration 48: Average Return = 107.98
Iteration 49: Average Return = 108.98
Iteration 50: Average Return = 109.7
Iteration 51: Average Return = 127.51
Iteration 52: Average Return = 133.49
Iteration 53: Average Return = 134.71
Iteration 54: Average Return = 144.55
Iteration 55: Average Return = 160.76
Iteration 56: Average Return = 155.88
Iteration 57: Average Return = 163.77
Iteration 58: Average Return = 168.98
Iteration 59: Average Return = 176.08
Iteration 60: Average Return = 184.91
Iteration 61: Average Return = 173.14
Iteration 62: Average Return = 180.68
Iteration 63: Average Return = 184.49
Iteration 64: Average Return = 174.42
Iteration 65: Average Return = 191.2
Iteration 66: Average Return = 172.98
Iteration 67: Average Return = 181.07
Iteration 68: Average Return = 188.2
Iteration 69: Average Return = 186.61
Iteration 70: Average Return = 184.2
Iteration 71: Average Return = 190.2
Iteration 72: Average Return = 194.13
Iteration 73: Average Return = 188.85
Iteration 74: Average Return = 183.23
Iteration 75: Average Return = 185.93
Iteration 76: Average Return = 190.68
Iteration 77: Average Return = 195.47
Solve at 77 iterations, which equals 7700 episodes.

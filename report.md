# Homework3-Policy-Gradient report

TA: try to elaborate the algorithms that you implemented and any details worth mentioned.

Overview
In this assignment, we solve a classic control problem - CartPole using policy gradient methods.

Problem 1: Construct a neural network to represent policy

code:

        hidden_layer = tf.layers.dense(self._observations, hidden_dim, activation=tf.tanh)
        probs = tf.layers.dense(hidden_layer, out_dim, activation=tf.nn.softmax)

Problem 2: Compute the surrogate loss

 <p align="center"><img src="images/3_2.PNG" width=50%/></p>
 
code:

         surr_loss = -tf.reduce_mean(tf.multiply(log_prob,self._advantages))
 
 Problem 3: Reduce the variance of the gradient estimation
 
 Use baseline to reduce the variance of our gradient estimate.

 <p align="center"><img src="images/3_3.PNG" width=50%/></p>
 
code:

         a=r-b   //r is reward, b is the values predicted by our baseline
result:

Iteration 1: Average Return = 19.34
Iteration 2: Average Return = 20.15
Iteration 3: Average Return = 21.76
Iteration 4: Average Return = 23.64
Iteration 5: Average Return = 23.45
Iteration 6: Average Return = 24.98
Iteration 7: Average Return = 28.34
Iteration 8: Average Return = 27.64
Iteration 9: Average Return = 28.07
Iteration 10: Average Return = 33.04
Iteration 11: Average Return = 41.88
Iteration 12: Average Return = 39.24
Iteration 13: Average Return = 41.23
Iteration 14: Average Return = 38.01
Iteration 15: Average Return = 43.16
Iteration 16: Average Return = 43.86
Iteration 17: Average Return = 48.01
Iteration 18: Average Return = 46.57
Iteration 19: Average Return = 46.66
Iteration 20: Average Return = 52.73
Iteration 21: Average Return = 53.45
Iteration 22: Average Return = 54.92
Iteration 23: Average Return = 51.85
Iteration 24: Average Return = 55.17
Iteration 25: Average Return = 58.66
Iteration 26: Average Return = 60.45
Iteration 27: Average Return = 63.34
Iteration 28: Average Return = 67.86
Iteration 29: Average Return = 61.66
Iteration 30: Average Return = 64.87
Iteration 31: Average Return = 67.1
Iteration 32: Average Return = 67.74
Iteration 33: Average Return = 67.98
Iteration 34: Average Return = 69.34
Iteration 35: Average Return = 70.76
Iteration 36: Average Return = 80.16
Iteration 37: Average Return = 74.97
Iteration 38: Average Return = 84.54
Iteration 39: Average Return = 85.93
Iteration 40: Average Return = 91.22
Iteration 41: Average Return = 90.57
Iteration 42: Average Return = 106.0
Iteration 43: Average Return = 107.42
Iteration 44: Average Return = 104.91
Iteration 45: Average Return = 117.96
Iteration 46: Average Return = 117.71
Iteration 47: Average Return = 128.8
Iteration 48: Average Return = 130.62
Iteration 49: Average Return = 127.35
Iteration 50: Average Return = 133.28
Iteration 51: Average Return = 131.5
Iteration 52: Average Return = 140.51
Iteration 53: Average Return = 136.71
Iteration 54: Average Return = 146.16
Iteration 55: Average Return = 140.59
Iteration 56: Average Return = 150.96
Iteration 57: Average Return = 153.68
Iteration 58: Average Return = 152.97
Iteration 59: Average Return = 153.13
Iteration 60: Average Return = 159.27
Iteration 61: Average Return = 172.11
Iteration 62: Average Return = 167.04
Iteration 63: Average Return = 174.03
Iteration 64: Average Return = 174.89
Iteration 65: Average Return = 176.5
Iteration 66: Average Return = 187.8
Iteration 67: Average Return = 186.42
Iteration 68: Average Return = 187.91
Iteration 69: Average Return = 193.09
Iteration 70: Average Return = 195.37
Solve at 70 iterations, which equals 7000 episodes. 

<p align="center"><img src="images/3_3_1.PNG" width=50%/></p>
<p align="center"><img src="images/3_3_2.PNG" width=50%/></p>
 
Problem 4: Remove baseline

if rewards are all positive, all trajectory's probability will increase. So we can add baeline to minimizes the variance of the gradient estimate.They are by definition on-policy and need to forget data very fast in order to avoid the introduction of a bias to the gradient estimator

result:

baseline = LinearFeatureBaseline(env.spec)

<p align="center"><img src="images/3_3_1.PNG" width=50%/></p>
<p align="center"><img src="images/3_3_2.PNG" width=50%/></p>

baseline = None

<p align="center"><img src="images/3_4_1.PNG" width=50%/></p>
<p align="center"><img src="images/3_4_2.PNG" width=50%/></p>


Problem 5: Actor-Critic algorithm (with bootstrapping)
 We use the one-step bootstrap for the advantage function
 
 <p align="center"><img src="images/3_5.PNG" height=50%/></p>
 
 code:

      b_roll=np.roll(b,-1)
      b_roll[-1]=0
      y = x + discount_rate * b_roll
      return y

 
 Problem 6: Generalized Advantage Estimationwe 
 we use a novel advantage function called "Generalized Advantage Estimation", which introduces one hyperparameter  λ  to compromise the above two estimation methods.
 
 <p align="center"><img src="images/3_6.PNG" height=50%/></p>
 
 code:
      
         a = util.discount(a,self.discount_rate*LAMBDA)

result:
Iteration 1: Average Return = 31.35
Iteration 2: Average Return = 32.09
Iteration 3: Average Return = 32.83
Iteration 4: Average Return = 37.05
Iteration 5: Average Return = 46.4
Iteration 6: Average Return = 47.56
Iteration 7: Average Return = 49.19
Iteration 8: Average Return = 54.65
Iteration 9: Average Return = 57.94
Iteration 10: Average Return = 60.7
Iteration 11: Average Return = 61.32
Iteration 12: Average Return = 58.53
Iteration 13: Average Return = 69.33
Iteration 14: Average Return = 74.87
Iteration 15: Average Return = 86.5
Iteration 16: Average Return = 89.0
Iteration 17: Average Return = 88.85
Iteration 18: Average Return = 91.18
Iteration 19: Average Return = 95.63
Iteration 20: Average Return = 103.54
Iteration 21: Average Return = 106.22
Iteration 22: Average Return = 109.16
Iteration 23: Average Return = 116.28
Iteration 24: Average Return = 128.67
Iteration 25: Average Return = 129.61
Iteration 26: Average Return = 139.21
Iteration 27: Average Return = 138.64
Iteration 28: Average Return = 136.64
Iteration 29: Average Return = 139.85
Iteration 30: Average Return = 148.68
Iteration 31: Average Return = 158.7
Iteration 32: Average Return = 161.94
Iteration 33: Average Return = 162.39
Iteration 34: Average Return = 166.93
Iteration 35: Average Return = 171.35
Iteration 36: Average Return = 175.21
Iteration 37: Average Return = 170.43
Iteration 38: Average Return = 171.96
Iteration 39: Average Return = 180.71
Iteration 40: Average Return = 181.05
Iteration 41: Average Return = 177.19
Iteration 42: Average Return = 176.23
Iteration 43: Average Return = 180.91
Iteration 44: Average Return = 178.15
Iteration 45: Average Return = 184.52
Iteration 46: Average Return = 177.26
Iteration 47: Average Return = 187.64
Iteration 48: Average Return = 186.0
Iteration 49: Average Return = 185.44
Iteration 50: Average Return = 182.33
Iteration 51: Average Return = 187.26
Iteration 52: Average Return = 189.36
Iteration 53: Average Return = 186.02
Iteration 54: Average Return = 189.86
Iteration 55: Average Return = 190.01
Iteration 56: Average Return = 188.46
Iteration 57: Average Return = 195.14
Solve at 57 iterations, which equals 5700 episodes.

<p align="center"><img src="images/6.PNG" width=50%/></p>

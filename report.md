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

Solve at 57 iterations, which equals 5700 episodes.

<p align="center"><img src="images/6.PNG" width=50%/></p>

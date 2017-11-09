# Homework3-Policy-Gradient report

TA: try to elaborate the algorithms that you implemented and any details worth mentioned.

Problem 1 ~ 3

Without the baseline, the probability of path decision will end up a bad result. In this case, we define a =r-b so 
if the reward is bigger than the baseline prediction, the probability will increase.

Code EX:
        NN = tf.layers.dense(inputs=self._observations, units=hidden_dim, activation=tf.nn.tanh)
        probs = tf.layers.dense(inputs=NN, units=out_dim, activation=tf.nn.softmax)
        
Code EX:
surr_loss = -tf.reduce_mean(tf.multiply(log_prob, self._advantages))



Problem 4

In this problem, we replace the baseline linearFeatureBseline with none.
And, without the baseline, we can see that the variation is bigger than the previous experiment. Therefore, the loss seems to converge more difficultly.

Code EX:
Baseline = None

Problem 5,6

In the previous problem, we use the baseline to reduce the variance. In this problem, we are going to use the one-step bootstrap to solve the problem. We will make each timestep's reward add on the multiplication of discount_rate and baseline prediction to make it the new r(reward). For problem 6, we just simply introduce the GAE which will use lambda to compromise the above methods.

Code EX:
    B = np.append(b[1:],0) 
    y = x+discount_rate*B
    return y



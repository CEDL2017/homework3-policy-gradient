# Homework3-Policy-Gradient report

TA: try to elaborate the algorithms that you implemented and any details worth mentioned.
## Problem 1

    `hiddenLayer = tf.contrib.layers.fully_connected( inputs = self._observations, num_outputs = hidden_dim, 
                                                 activation_fn = tf.nn.tanh )
        probs = tf.contrib.layers.fully_connected( inputs = hiddenLayer, num_outputs = out_dim, 
                                                activation_fn = tf.nn.softmax )`

## Problem 2

    `surr_loss = -tf.reduce_mean(log_prob*self._advantages)`
    
## Problem 3

    `a = r - b`
    
## Problem 5

    `b = np.append(b[1:], 0)
    return x + b * discount_rate`

## Problem 6
    
    `a = util.discount(a, self.discount_rate * LAMBDA)`
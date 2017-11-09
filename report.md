# Homework3-Policy-Gradient report

## problem1 construct a neural network to represent policy
Added two fully connected layers , where the first fc layer has 'hidden_dim' output dimension and the second fc layer has 'out_dim' dimension

## problem2 compute the surrogate loss
surrogate loss is to maximize the log likelyhood, so we add negative sign to minimize -log likelyhood.

## problem3 
the reward of advantage is total reward minus baseline

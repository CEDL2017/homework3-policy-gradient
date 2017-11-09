# Homework3-Policy-Gradient report


### 105061528 陳玉亭


### Overview:
>This homework is about policy gradient. Based on policy gradient, Actor-Critic algorithm and Generalized Advantage Estimation (GAE) methods are  extensions.

#### For problem.4, compare reducing baseline results wih not do this:
**Without baseline: </br>**
<img src="imgs/loss_base.png"  width="300px">
<img src="imgs/return_base.png"  width="300px"> </br>
**With reducing baseline: </br>**
<img src="imgs/loss_reduce.png"  width="300px">
<img src="imgs/return_reduce.png"  width="300px"> </br>
See lower variance in reducing baseline results (smooth curves). For reducing baseline, it achieve successful performance early (from Retrun figures).</br>

**explain why the baseline won't introduce bias:** </br>
If we take gradient on the policy loss with baseine reduction, we can observe: </br>
<img src="imgs/add_baseline.JPG"  width="300px"> </br>
We know the first term grad(logP*R) has no bias. Then, for the last term, that can be derived: </br>
<img src="imgs/derived.JPG"  width="300px">, which shows no bias included.

 



  



 

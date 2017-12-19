

# Homework3-Policy-Gradient report

前兩題分別把 Policy Gradient的網路架構和loss function造出

## Problem 3: Use baseline to reduce the variance of our gradient estimate
前3題主要在架設Policy-Gradient的結構  
運用神經網路計算出各方向的機率做加分  
越往上(陡峭)分數越高  
然而要設定baseline使得較差的路徑分數下降
![Q3](https://github.com/w95wayne10/homework3-policy-gradient/blob/master/photo/Q3.png)

## Problem 4  
測試拿掉baseline的狀況  
收斂速度下降  
![Q4](https://github.com/w95wayne10/homework3-policy-gradient/blob/master/photo/Q4.png)

## Problem 5 Actor-Critic algorithm (with bootstrapping)
## Problem 6: Generalized Advantage Estimation
從採樣方式(5題)  
和baseline取法(各方向期望值)做優化(6題)  
使之更快速完成訓練  
從圖中可看出收斂速度明顯上升  
(因為是結合第三題和第五題的結果 如第五題寫錯會反映在此題)
![Q6](https://github.com/w95wayne10/homework3-policy-gradient/blob/master/photo/Q6.png)

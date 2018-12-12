# lbt
## Milestone

2018.11.16 Fix the initial value assignment bug of accu_value and reminder in Dense layer. The best accuracy 89.11%.

2018.11.23 Linear Quantization(acc 87.4%, ~Pure DFXP)

2018.11.29 Change initial lr to 2e-3, stable accuracy for lazy_update is 89.1%(best 89.50%)

2018.12.08 Add GradBuffer to first conv layer, now best stable accuracy 89.4%(best 89.57%)

2018.12.10 Using uniform distribution stochastic in grad, model converge at lr=5e-3, stable acc ~90.0%

## To be verified

2018.11.28 Linear_q, Quantize grad. in Dense layer on channel axis(10 respectively)
(87.4% -> 88.0%, not stable)

2018.11.29 Linear_q, Quantize weight in ALL conv layer on channel axis.

# IllegalInput Error:
the input should be the format of dataframe，and first column should be the label column!
输入数据需要是带列名的dataframe的格式，且第一列需要为index列，形如：
```
# ip	qps	udidCount	uidCount	deviceTypeCount
# 188.64.68.10	0.11666667	1	1	1
# 222.65.20.16	1.16	1	3	2
# 311.16.63.214	0.31333333	1	2	1
# 888.104.6.24	0.37666667	1	2	1
```

# IllegalInputSize Error:
The input data amount cannot match the required size,Pls increase input data amount or increase the max_limit!
计算切比雪夫不等式过程中数据量过小，或者max_limit过小，建议增加！

# Warnings ：The covariation of some column turns to be zero,it will be remove for calculating!
协方差计算过程中有一列为0

# Warnings ：Some columns seem to be approximate multicollinearity,it will be remove for calculating!
存在多列共线性，形如：
```
# qps	udidCount	uidCount	deviceTypeCount
# 0.11666667	1	2	1
# 1.16	1	2	1
# 0.31333333	1	2	1
# 0.37666667	1	2	1
```
import pandas as pd
from Frcwp import Frcwp

path = '../data/data_all.csv'
traindata = pd.read_csv(path)

'''
data like this :
                     ip        f1  f2  f3  f4  f5  f6
0      204.156.160.102   0.116667   1   1   3  23   0
1      205.157.112.108   1.160000   1   3   4  59   3
2      205.108.155.306   0.313333   1   2   3  43   7
'''

# we should change the dataframe like this:
frc = Frcwp()
traindata = frc.changeformat(traindata, index=0)
'''
change the columns=0 as the index , the f1~f6 could be the calculation feature
                        f1  f2  f3  f4  f5  f6
ip
204.156.160.102   0.116667   1   1   3  23   0
205.157.112.108   1.160000   1   3   4  59   3
205.108.155.306   0.313333   1   2   3  43   7
'''

# You can define your own outlier size , the details of these params can be got from ../Frcwp/Frcwp.py:
params = {
    'na_rate': 0.4,
    'single_dealed': 1,
    'is_scale': 0,
    'distince_method': 'Maha',
    'outlier_rate': 0.05,
    'strange_rate': 0.15,
    'nestimators': 150,
    'contamination': 0.2
}

# train the frc model
frc.fit(traindata, **params)

# you can get the following attributes:
# 1.useful_index
# you can get which column_rank used in trained model
frc.useful_index
# Out[47]: [0, 1, 2, 3, 4, 5]

# 2.normalbarycentre
# you can get common data in features
frc.normalbarycentre
# Out[48]:
# f1     0.364080
# f2     1.044306
# f3     1.451814
# f4     3.039597
# f5    31.752073
# f6     2.402335
# dtype: float64

# 3.potentialdata_set
# you can get potential outlier data suggested by model , but here ,you can ignore it with your own outliers as well
frc.potentialdata_set

# You can define your own potential outliers, the details of these params can be got from ../Frcwp/Frcwp.py:
# predict outliers with the trained frc model
predict_params = {
    'output': 20,
    'is_whole': 1
}
frc.predict(frc.potentialdata_set, **predict_params)

# if you want get the whole probability of your potential outliers
frc.similarity_label

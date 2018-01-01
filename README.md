![](https://img.shields.io/badge/license-MIT-000000.svg)
# What is Frcwp
It means fast risk control with python.
It's a lightweight tool that automatic recognize the outliers from a large data pool. 
This project aims to help people get easily method with abnormal recognition, especially forces password attacks.
We wish it could be a nice Open Source which could simplify the complexity of the Data Feature Project.  

# Theory
**@bolg:[风控用户识别方法](http://shataowei.com/2017/12/09/风控用户识别方法/)**

# Compared with common methods
We got the correctness around 29 data sets below,however the speed of Frcwp comes last. 
![](http://upload-images.jianshu.io/upload_images/1129359-90b9e7933f787fd4.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

# Usage
U can get it easily download from Pypi with **`pip install Frcwp`**.

```python
import pandas as pd
from Frcwp import Frcwp

path = '../data/data_all.csv'
traindata = pd.read_csv(path)

frc = Frcwp()
traindata = frc.changeformat(traindata, index=0)

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

frc.fit(traindata, **params)

predict_params = {
    'output': 20,
    'is_whole': 1
}
frc.predict(frc.potentialdata_set, **predict_params)
```

# Dependence
Frcwp is implemented in Python 3.6, use Pandas.DataFrame to store data. These package can be easily installed using pip.
- [pandas](https://github.com/pandas-dev/pandas)
- [numpy](https://github.com/numpy/numpy)
- [sklearn](http://scikit-learn.org/stable/documentation.html)

# Reference
- [数据预处理-异常值识别](http://shataowei.com/2017/08/09/数据预处理-异常值识别/)
- [多算法识别撞库刷券等异常用户](http://shataowei.com/2017/12/01/多算法识别撞库刷券等异常用户/)

# TODO
- feature scanning
- increase new outliers distinguishing methods

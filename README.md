![](https://img.shields.io/badge/license-MIT-000000.svg)
# What is Frcwp?
It means fast risk control with python.It's a lightweight tool that automatic recognize the outliers from a large data pool. ALL u need to do is making the data into a dataframe with columns. This project aims to help people get easily method with abnormal recognition, especially forces password attacks.We wish it could be a nice Open Source which could simplify the complexity of the Data Feature Project.  

# Theory
**@bolg:[风控用户识别方法](http://shataowei.com/2017/12/09/风控用户识别方法/)**

# Compared with common ways
We got the correctness around 29 data sets below,however the speed of Frcwp comes last. 
![](http://upload-images.jianshu.io/upload_images/1129359-90b9e7933f787fd4.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

# Usage
U can get it easily download from [pypi](https://pypi.python.org/pypi?:action=display&name=fast_risk_control&version=0.0.1)
or u can also **`pip install Frcwp`**.
```
#加载包
from Frcwp import Frcwp

#使用方法
Frcwp.transform(data)
```

# Dependence
Frcwp is implemented in Python 3.6, use Pandas.DataFrame to store data. These package can be easily installed using pip.
- [isolation forest地址](https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/ensemble/iforest.py)
- [pandas、numpy下载地址](http://www.lfd.uci.edu/~gohlke/pythonlibs/)

# Reference：
- [数据预处理-异常值识别](http://shataowei.com/2017/08/09/数据预处理-异常值识别/)
- [多算法识别撞库刷券等异常用户](http://shataowei.com/2017/12/01/多算法识别撞库刷券等异常用户/)

# Release
- V0.0.2 : increase some function.
- V0.0.3 : increase the method at small data.
- V0.0.4 : rename the package

# TO-DO
- feature scanning
- increase new outliers distinguishing methods

# What is it?
It is a linear model, based on the crossed parameters. This project aims to help people get deeper insights into FM, especially Crossed Parameters Training. 

# Theory
**@bolg:[FM](http://shataowei.com/2017/12/04/FM理论解析及应用/)**

￼![](http://upload-images.jianshu.io/upload_images/1129359-92da0691440d9857.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

# Overview
#### 1.fastfm
Train by the package `fastfm` directly at fastfm_demo.
It is important to note that the input of `fastfm` should be sparse data. If u got the large amount of continues features,you'll be trouble with the data preprocessing.

#### 2.Fsfm
We rewrite fm by ourselves and focus helping people get deeper insights about FM.So we upload it to the pypi named 'Fsfm',you can downlode it if you're interested in it.
We use the sgd in solving the problems. The good news is we can use the continuous variable directly but the speed is a trouble.

# Dependencies
- pandas
- numpy
- math
- sklearn.preprocessing

# Useage:
**pip install Fsfm**

```
from Fsfm import fm
fm = fm()
#fm.fit()
#fm.predict()
```

U can get more from the *example* folder.

# TODO
- speed up training and predicting
- get it more official,now it's still just a demo-like. 

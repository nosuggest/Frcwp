import pandas as pd
import numpy as np


class distince():
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y
        assert self.x.shape == self.y.shape, '输入数据格式不一致'

    def euclidean_dist(self):
        return np.linalg.norm(self.x - self.y)

    def Mahalanobis_dist(self, Xtrain=None):
        try:
            Xtrain_Cov = np.cov(Xtrain.T)
            Xtrain_Inv = np.linalg.inv(Xtrain_Cov)
        except:
            raise ValueError()
        delta = self.x - self.y
        return np.sqrt(np.dot(np.dot(delta, Xtrain_Inv), delta.T))


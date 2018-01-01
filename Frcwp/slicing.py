import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


class grouped():
    def __init__(self, X=None, similarity=None, outlier_rate=0.01, strange_rate=0.1):
        self.X = X
        self.similarity = pd.DataFrame(similarity)
        self.similarity.index = self.X.index
        self.outlier_rate = outlier_rate
        self.strange_rate = strange_rate

    def outlier_group(self):
        ogroup = pd.concat([self.similarity, self.X], axis=1)
        ogroup.columns = ['simi'] + list(ogroup.columns[1:])
        ogroup = ogroup.sort_values(by='simi', ascending=False)
        outliercutoff = int(self.outlier_rate * self.X.shape[0]) + 1
        return ogroup.iloc[:outliercutoff, :], outliercutoff

    def strange_group(self):
        sgroup = pd.concat([self.similarity, self.X], axis=1)
        sgroup.columns = ['simi'] + list(sgroup.columns[1:])
        sgroup = sgroup.sort_values(by='simi', ascending=False)
        _, outliercutoff = self.outlier_group()
        strangecutoff = int(self.strange_rate * self.X.shape[0])
        return sgroup.iloc[outliercutoff:outliercutoff + strangecutoff, :], strangecutoff

    def common_group(self):
        commongroup = pd.concat([self.similarity, self.X], axis=1)
        commongroup.columns = ['simi'] + list(commongroup.columns[1:])
        commongroup = commongroup.sort_values(by='simi', ascending=False)
        _, outliercutoff = self.outlier_group()
        _, strangecutoff = self.strange_group()
        return commongroup.iloc[(outliercutoff + strangecutoff):, :]


class isolationforest():
    def __init__(self, X=None, nestimators=None, contamination=None):
        self.X = X
        self.nestimators = nestimators
        self.contamination = contamination
        self.ift = None

    def fit(self):
        isf = IsolationForest(n_estimators=self.nestimators, contamination=self.contamination)
        isf.fit(self.X)
        self.ift = isf

    def predict(self, pred_data):
        # model = self.fit()
        assert self.X.shape[1] == pred_data.shape[1], '预测数据与训练数据格式不一致～'
        score = self.ift.decision_function(pred_data)
        return score


class iswhole():
    def __init__(self, X=None):
        self.X = X

    def getoutput(self, initaldata):
        wholeoutindex = [x for x in set(self.X.index)]
        return initaldata.ix[wholeoutindex, :]

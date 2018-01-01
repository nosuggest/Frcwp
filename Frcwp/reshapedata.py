import pandas as pd
import numpy as np


def formatcheck(X):
    if isinstance(X, pd.DataFrame) != 1:
        X = pd.DataFrame(X)
    # X.reset_index(drop=True, inplace=True)
    return X


class natest():
    def __init__(self, X=None, na_rate=0.4):
        assert na_rate > 0, 'na_rate不能小于等于0'
        assert na_rate < 1, 'na_rate不能大于等于1'
        self.X = formatcheck(X)
        self.na_rate = na_rate

    def naremove(self):
        _to_remove_columns = []
        for i in range(self.X.shape[1]):
            if np.isnan(self.X.iloc[:, i]).sum() >= self.na_rate * self.X.shape[0]:
                _to_remove_columns.append(i)
        _keep_columns = [x for x in range(self.X.shape[1]) if x not in _to_remove_columns]
        return self.X.iloc[:, _keep_columns]


class valuenumber():
    def __init__(self, X=None):
        self.X = X

    def singlevalueremove(self):
        _to_remove_columns = []
        for i in range(self.X.shape[1]):
            if len(set(self.X.iloc[:, i])) == 1:
                _to_remove_columns.append(i)
        _keep_columns = [x for x in range(self.X.shape[1]) if x not in _to_remove_columns]
        return self.X.iloc[:, _keep_columns]


class standardizeddata():
    def __init__(self, X=None):
        self.X = formatcheck(X)

    def standardstep(self):
        return (self.X - self.X.min()) / (self.X.max()-self.X.min())


class coltest():
    def __init__(self, X=None):
        self.X = X

    def zerotest(self):
        _zerocolumns = []
        for i in range(self.X.shape[1]):
            if abs(self.X.iloc[:, i]).sum() == 0:
                _zerocolumns.append(i)
        _no_zerocolumns = [x for x in range(self.X.shape[1]) if x not in _zerocolumns]
        return self.X.iloc[:, _no_zerocolumns]

    def columnstest(self):
        _collinearity = []
        collinearity_data = self.zerotest()
        for i in range(self.zerotest().shape[1]):
            for j in range(self.zerotest().shape[1]):
                # cal the half of the whole data
                if j > i:
                    Li = np.linalg.norm(collinearity_data.iloc[:, i])
                    Lj = np.linalg.norm(collinearity_data.iloc[:, j])
                    Lij = np.dot(collinearity_data.iloc[:, i], collinearity_data.iloc[:, j])
                    # assert the zero vector and ssert the collinearity vector
                    if (abs(Lij - (Li * Lj)) < 1e-4):
                        _collinearity.append(i)
                        break
        _no_collinearity = [x for x in range(collinearity_data.shape[1]) if x not in _collinearity]
        return collinearity_data.iloc[:, _no_collinearity], _no_collinearity

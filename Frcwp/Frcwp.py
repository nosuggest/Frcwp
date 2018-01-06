import pandas as pd
import numpy as np
import sys
from .reshapedata import natest as _natest
from .reshapedata import valuenumber, coltest, standardizeddata, formatcheck
from .distince import distince
from .slicing import grouped, isolationforest, iswhole


class Frcwp():
    '''
    param : na_rate : if na_rate != None , remove some column if the column with too many nan values
    param : single_dealed : if single_dealed > 0 , remove some column if the column with single value
    param : is_scale : if is_scale = 1 , scale the data to improve the calculation speed ,may change the distribution of the initial data , is_scale =0 may be better usually
    param : distince_method : if 'Maha' then Mahalanobis_dist ; if 'Eucl' then euclidean_dist
    param : outlier_rate : the estimated outliers / all cases , the smaller, the better but not smaller than 0
    param : strange_rate : the strange outliers(潜在可能的异常点) / all cases
    param : nestimators : isolation tree number
    param : contamination : actual estimated outliers / all cases
    param : is_whole : if is_whole = 1 , the output features is the same as input ; if is_whole = 0 ,the output features are the features which take part in the training process
    param : output : if None then output all the potentialdata , if 0<output<1 then treated as rate , if output>1 the treated as number

    attribute : useful_index : the feature rank_index used for the model training actually
    attribute : similarity_label : the output data outlier degree the larger the outlier degree higher
    attribute : normalbarycentre : common level of your input data
    attribute : potentialdata_set : the suggested outlier potential data set , you can use your outlier potential data as well

    '''

    def __init__(self):
        self.na_rate = None
        self.single_dealed = None
        self.is_scale = None
        self.distince_method = None
        self.outlier_rate = None
        self.strange_rate = None
        self.nestimators = None
        self.contamination = None
        self.is_whole = None
        self.output = None
        self.iforest = None
        self.useful_index = None
        self.original_data = None
        self.similarity_label = None

    def changeformat(self, X, index=0):
        assert isinstance(index, int), '请输入识别列的列序号，以0开始'
        if isinstance(X, pd.DataFrame) != 1:
            X = formatcheck(X)
        X.index = X.iloc[:, index]
        keep_index = [x for x in range(X.shape[1]) if x != index]
        return X.iloc[:, keep_index]

    def fit(self, X=None, na_rate=None, single_dealed=None, is_scale=0, distince_method='Maha', outlier_rate=0.01,
            strange_rate=0.1, nestimators=100, contamination=0.1):
        self.na_rate = na_rate
        self.single_dealed = None
        self.is_scale = is_scale
        self.distince_method = distince_method
        self.outlier_rate = outlier_rate
        self.strange_rate = strange_rate
        self.nestimators = nestimators
        self.contamination = contamination
        self.normalbarycentre = None
        self.Metricset = None
        self.potentialdata_set = None
        self.original_data = X.copy()

        if isinstance(X, pd.DataFrame) != 1:
            print('we will change your data as the format of dataframe~')

        # begin preprocessing the data
        if self.na_rate != None:
            natt = _natest(X, self.na_rate)
            X = natt.naremove()

        if self.single_dealed:
            vnb = valuenumber(X)
            X = vnb.singlevalueremove()

        if self.is_scale:
            stdd = standardizeddata(X)
            X = stdd.standardstep()

        # begin outliers pre-recognition
        cal_X = X.copy()
        # remove the zero_columns and the collinearity_data
        colt = coltest(cal_X)
        colt_cal_X, colt_col_index = colt.columnstest()
        cov_cal_X = np.cov(colt_cal_X.T)
        self.useful_index = colt_col_index

        if self.distince_method not in ['Maha', 'Eucl']:
            raise NotImplementedError('distince_method should be Maha or Eucl~')

        if self.distince_method == 'Maha':
            colt1 = coltest(pd.DataFrame(cov_cal_X))
            colt1_cal_X, colt1_col_index = colt1.columnstest()
            if len(colt1_col_index) <= 1:
                raise ValueError(
                    'the outlier among the train data is too small ,PLS turn the is_scale = 0 or add reshape data')
            while cov_cal_X.shape != colt1_cal_X.shape:
                colt = coltest(pd.DataFrame(colt_cal_X).iloc[:, colt1_col_index])
                colt_cal_X, colt_col_index = colt.columnstest()
                cov_cal_X = np.cov(colt_cal_X.T)
                colt1 = coltest(cov_cal_X)
                colt1_cal_X, colt1_col_index = colt1.columnstest()

            cal_X_colt = cal_X.iloc[:, colt1_col_index]
            normalbarycentre = cal_X_colt.mean(axis=0)

            # calculate each case normal degree
            similarity_d = []
            for i in range(cal_X_colt.shape[0]):
                dist = distince(cal_X_colt.iloc[i, :], normalbarycentre)
                similarity_d.append(dist.Mahalanobis_dist(cal_X_colt))
        else:
            normalbarycentre = colt_cal_X.mean(axis=0)
            similarity_d = []

            for i in range(colt_cal_X.shape[0]):
                dist = distince(colt_cal_X.iloc[i, :], normalbarycentre)
                similarity_d.append(dist.euclidean_dist())
        self.normalbarycentre = normalbarycentre
        # spilt all user into outlier,strange and common part
        ggp = grouped(colt_cal_X, similarity_d, self.outlier_rate, self.strange_rate)
        outlierset, _ = ggp.outlier_group()
        strangeset, _ = ggp.strange_group()
        commonset = ggp.common_group()

        traindata = pd.concat([outlierset, commonset], axis=0)
        potentialdata = pd.concat([outlierset, strangeset], axis=0)
        traincol = [x for x in traindata.columns if x != 'simi']
        potentialcol = [x for x in potentialdata.columns if x != 'simi']
        self.Metricset = traindata[traincol]
        self.potentialdata_set = potentialdata[potentialcol]
        # score the cases in outlier and strange part
        ift = isolationforest(self.Metricset, self.nestimators, self.contamination)
        ift.fit()
        self.iforest = ift

    def predict(self, potentialdata, output, is_whole):
        potentialdata = potentialdata.copy()
        self.is_whole = is_whole
        self.output = output
        score = pd.DataFrame(self.iforest.predict(potentialdata))
        score.index = potentialdata.index
        potentialdata['simi'] = score

        self.similarity_label = (abs(potentialdata['simi']) - abs(potentialdata['simi']).min()) / (
            abs(potentialdata['simi']).max() - abs(potentialdata['simi']).min())

        if self.output == None:
            potentialdata = potentialdata.sort_values(by='simi')
        elif self.output > 1:
            potentialdata = potentialdata.sort_values(by='simi')
            potentialdata = potentialdata.iloc[:self.output, :]
        elif self.output > 0 and self.output < 1:
            potentialdata = potentialdata.sort_values(by='simi')
            assert (self.output * self.original_data.shape[0]) < potentialdata.shape[
                0], '你想要产出的异常点超过预估点，请降低异常点数output值'
            potentialdata = potentialdata.iloc[:int(self.output * self.original_data.shape[0]), :]

        assert abs(potentialdata['simi']).max() != abs(potentialdata['simi']).min(), '数据无明显离散异常点'

        # output
        if self.is_whole:
            isw = iswhole(potentialdata)
            out = isw.getoutput(self.original_data)
        else:
            outindex = [x for x in potentialdata.columns if x != 'simi']
            out = potentialdata[outindex]
        return out

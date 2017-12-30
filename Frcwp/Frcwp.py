# -*- coding: utf-8 -*-
try:
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import IsolationForest
    import sys
    import logging
except:
    print('缺少前置的包或对应包的版本不一致，请在readme.md中核对！')

'''
    Develop_information
    ----------
    __time__ = '20170929'
    __author__ = 'sladesal'
    __blog__ = 'www.shataowei.com'

    Parameters
    ----------
    train_data_res ： 需要识别的数据
    max_limit ： ～(0,0.5)，切比雪夫不等式中的容忍度，异常数据筛选的严格程度，越小则越严格，筛选的预训练的异常数据越少
    nestimators : ～(50,150)，孤立森林数个树
    contamination : ～(0,0.1)， 预估异常数据占比
    max_output ： 最大允许产出的异常数据的条数

    Attributes
    ----------
    heavy_point : 正常用户的各feature的状态，用来对比异常数据的结果
    outlier_details : 识别出来的outlier及对应的feature值，prob越小代表越异常
    clf : isolation forest模型
    score : 潜在异常用户的得分

    References
    ----------
    .. [0] https://zh.wikipedia.org/wiki/切比雪夫不等式
    .. [1] Liu, Fei Tony, Ting, Kai Ming and Zhou, Zhi-Hua. "Isolation forest."
           Data Mining, 2008. ICDM'08. Eighth IEEE International Conference on.
    .. [2] https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/ensemble/iforest.py
'''


class Frcwp(object):
    def __init__(self):
        self.train_data_res = None
        self.max_limit = None
        self.nestimators = None
        self.contamination = None
        self.max_output = None
        self.heavy_point = None  # 定义重心集合
        # self.score = None
        self.outlier_data_output = None
        self.outlier_details = None

    def transform(self, train_data_res, max_limit=0.005, nestimators=100, contamination=0.004,
                  max_output=15):
        try:
            drop_columns = [0]
            self.train_data_res = train_data_res
            try:
                self.train_data_res.iloc[0, 0]
            except:
                raise IllegalInput(
                    'Error : the input should be the format of dataframe，and first column should be the label column!')
                sys.exit(1)
            # feature个数
            max_col = self.train_data_res.shape[1]
            # 切比雪夫不等式切分数据集合
            # 去除类似outlier index（ip）的非计算指标
            if drop_columns is not None:
                real_columns = [x for x in range(max_col) if x not in drop_columns]
            else:
                real_columns = range(max_col)
            DF_G = self.train_data_res.iloc[:, real_columns]
            # 保存原始变量
            DF_G_B = DF_G.copy()
            DF_G_B_T = np.array(DF_G_B).T
            DF_G_B = np.array(DF_G_B)
            DF_G_B_T_new = DF_G_B_T
            DF_G_B_new = DF_G_B
            # 计算协方差矩阵及逆矩阵
            S = np.cov(DF_G_B_T_new)
            S_f = pd.DataFrame(S)

            # 判断协方差某两列一致
            badindex1 = []
            for i in range(S_f.shape[0]):
                for j in range(S_f.shape[0]):
                    if j > i:
                        if (max(sum(S_f[i]), sum(S_f[j])) % min(sum(S_f[i]), sum(S_f[j]))) == 0:
                            badindex1.append(j)
            # 如果协方差矩阵某列全为0，则删除该列的feature，重新计算协方差
            if (len(S_f[(S_f == 0).all()].index)) or (len(badindex1) > 0):
                while len(S_f[(S_f == 0).all()].index) > 0:
                    self.get_warnings()
                    print('Warnings ：The covariation of some column turns to be zero,it will be remove for calculating!')
                    badindex = S_f[(S_f == 0).all()].index
                    indexnew = [i for i in range(S_f.shape[0]) if i not in badindex]
                    # 如果独立变量小于二则无法检验，跳出异常检验，返回无异常用户
                    if len(indexnew) <= 1:
                        raise IllegalInputSize('error:the input data does not satisfy the limitation!')
                        sys.exit(1)
                    S_f = pd.DataFrame(np.cov(pd.DataFrame(DF_G_B_new[:, indexnew].T))).copy()
                # 剔除相关向量
                if len(badindex1) > 0:
                    self.get_warnings()
                    print(
                        'Warnings ：Some columns seem to be approximate multicollinearity,it will be remove for calculating!')
                    indexnew1 = [i for i in range(S_f.shape[0]) if i not in badindex1]
                    S_f = pd.DataFrame(np.cov(pd.DataFrame(DF_G_B_new[:, indexnew][:, indexnew1].T))).copy()
                    DF_G_B_new = DF_G_B_new[:, indexnew][:, indexnew1]
                    DF_G_B_T_new = np.array(DF_G_B_new).T
                    t_indexnew = indexnew
                    t_indexnew.insert(0, -1)
                    t_indexnew = list(np.array(t_indexnew) + 1)
                    t_indexnew1 = indexnew1
                    t_indexnew1.insert(0, -1)
                    t_indexnew1 = list(np.array(t_indexnew1) + 1)
                    self.train_data_res = self.train_data_res.iloc[:, t_indexnew].iloc[:, t_indexnew1]
                else:
                    DF_G_B_new = DF_G_B_new[:, indexnew]
                    DF_G_B_T_new = np.array(DF_G_B_new).T
                    t_indexnew = indexnew
                    t_indexnew.insert(0, -1)
                    t_indexnew = list(np.array(t_indexnew) + 1)
                    self.train_data_res = self.train_data_res.iloc[:, t_indexnew]
            S = S_f
            # 计算逆矩阵
            SI = np.linalg.inv(S)
            # 求全量数据的重心的距离
            DF_G_B_T_heavypoint = DF_G_B_T_new.mean(axis=1)
            d1 = []
            n = DF_G_B_new.shape[0]
            # 计算切比雪夫不等式中的马氏距离
            for i in range(n):
                delta = DF_G_B_new[i] - DF_G_B_T_heavypoint
                d = np.sqrt(np.dot(np.dot(delta, SI), delta.T))
                d1.append(d)
            # 异常用户集合，初筛千分之5
            d2 = pd.Series(d1)
            N = DF_G_B_new.shape[1]
            self.max_limit = max_limit
            pr = self.max_limit
            limit = np.sqrt(N / pr)
            outlier = d2[d2 > limit]
            # 防止抛空的异常用户,至少满足20个人的异常用户，并放入outlier_data异常集合中
            times_count = 0
            while len(outlier) < max(int(0.01 * n), 20):
                times_count = times_count + 1
                pr = pr + 0.005
                limit = np.sqrt(N / pr)
                outlier = d2[d2 > limit]
                if times_count > 100000:
                    raise IllegalInputSize(
                        'Error : The input data amount cannot match the required size,Pls increase input data amount or increase the max_limit')
                    sys.exit(1)
            index = outlier.index
            outlier_data = DF_G_B_new[index]
            outlier_data = np.array([I for I in outlier_data if ((I - DF_G_B_T_heavypoint) > 0).any()])
            # 防止抛空的未知用户，计算未知用户和正常用户,至少满足50个人的未知用户，并放入outlier_data1未知集合中
            N1 = N
            pr1 = pr + 0.005
            limit1 = np.sqrt(N1 / pr1)
            outlier1 = d2[d2 > limit1]
            times_count1 = 0
            while len(outlier1) < max(int(0.15 * n), 50):
                times_count1 = times_count1 + 1
                pr1 = pr1 + 0.005
                limit1 = np.sqrt(N1 / pr1)
                outlier1 = d2[d2 > limit1]
                if times_count1 > 100000:
                    raise IllegalInputSize(
                        'Error : The input data amount cannot match the required size,Pls increase input data amount or increase the max_limit')
                    sys.exit(1)
            index1 = outlier1.index
            index1 = [I for I in index1 if I not in index]
            # outlier_data1为未知用户
            outlier_data1 = DF_G_B_new[index1]
            outlier_data1 = np.array([I for I in outlier_data1 if ((I - DF_G_B_T_heavypoint) > 0).any()])
            if len(outlier_data1) == 0:
                self.get_warnings()
                print('Warnings ：For the info given by user,cannot get the unknown users!Pls check the input amount!')
            # 全量用户去除异常用户和未知用户，即为正常用户
            index2 = [i for i in range(n) if i not in index and i not in index1]
            common_data = DF_G_B_new[index2]
            self.heavy_point = common_data.mean(axis=0)
            # 如果没有数据，则返回没有未知用户，以异常用户代替未知用户+异常用户
            if len(outlier_data1) == 0:
                train = common_data
            else:
                train = np.r_[common_data, outlier_data]  # 训练数据合并
            # 预估的异常用户占比：contamination，参数初步定为千分之4。训练的树的个数：nestimators
            self.nestimators = nestimators
            self.contamination = contamination
            self.clf = IsolationForest(n_estimators=self.nestimators, contamination=self.contamination)
            self.clf.fit(train)
            # 如果没有数据，则返回没有未知用户，以异常用户代替未知用户+异常用户
            if len(outlier_data1) == 0:
                pre_format_train_data = outlier_data
            else:
                pre_format_train_data = np.r_[outlier_data, outlier_data1]
            # 计算得分并排序
            score = self.clf.decision_function(pre_format_train_data)
            K = np.c_[score, pre_format_train_data]
            k_rank = np.array(sorted(K, key=lambda x: x[0]))
            # 识别排序靠前contamination参数初步定为千分之4的3倍，也就是1.2%的用户为异常用户
            assume_rate = np.ceil(pre_format_train_data.shape[0] * self.contamination * 3)
            # 设置上限，最多产出预设的个风控用户
            self.max_output = max_output
            if assume_rate >= self.max_output:
                assume_rate = self.max_output
            outlier_data2 = k_rank[:int(assume_rate)]
            self.outlier_data_output = outlier_data2
            try:
                self.outlier_details = self.outlier_search(self.outlier_data_output, self.train_data_res)
            except:
                print('Error : you should run the transform first!')
                sys.exit(1)
        except:
            drop_columns = [0]
            self.train_data_res = train_data_res
            self.contamination = contamination
            max_col = self.train_data_res.shape[1]
            if drop_columns is not None:
                real_columns = [x for x in range(max_col) if x not in drop_columns]
            else:
                real_columns = range(max_col)
            DF_G = self.train_data_res.iloc[:, real_columns]
            DF_G = DF_G/DF_G.max(axis=0)
            self.heavy_point = DF_G.mean()/2
            self.heavy_point = self.heavy_point.T
            ouput_number = self.contamination * self.train_data_res.shape[0]
            ouput_number = max(ouput_number,1)
            res = pd.DataFrame()
            for i in range(DF_G.shape[0]):
                if (DF_G.iloc[i,:] - self.heavy_point).sum() > 0:
                    df = pd.concat([pd.DataFrame([np.linalg.norm(DF_G.iloc[i,:] - self.heavy_point)]),self.train_data_res.iloc[:, real_columns].iloc[i,:]],axis=0).T
                    res = pd.concat([res,df],axis=0)
            res = res.sort_values(0)
            self.outlier_data_output = pd.DataFrame()
            for i in range(ouput_number):
                index = res.shape[0] - i - 1
                self.outlier_data_output = pd.concat([self.outlier_data_output,res.iloc[index,:]],axis=0).T
            self.outlier_data_output = np.array(self.outlier_data_output)
            self.heavy_point = self.train_data_res.iloc[:, real_columns].mean()
            try:
                self.outlier_details = self.outlier_search(self.outlier_data_output, self.train_data_res)
            except:
                print('Error : you should run the transform first!')
                sys.exit(1)




    # 寻找outlier
    def outlier_search(self, outlier_dict, train_data_res):
        outlier = []
        values_res = pd.DataFrame()
        for i in range(outlier_dict.shape[0]):
            for k in range(train_data_res.shape[0]):
                if sum(train_data_res.iloc[k, 1:] - outlier_dict[i][1:]) == 0:
                    outlier_key0 = train_data_res.iloc[k, 0]
                    values = pd.DataFrame(outlier_dict[i]).T
                    values.columns = train_data_res.columns[1:].insert(0, 'prob')
                    outlier.append(outlier_key0)
                    values_res = pd.concat([values_res, values], axis=0)
            print('we have gotten the %s of %s outlier(s) already!' % (i + 1, outlier_dict.shape[0]))
        return outlier, values_res

    def get_warnings(self):
        logging.warning('This is warning message!')


class IllegalInput(Exception):
    """
    The input data format is wrong!
    """
    pass


class IllegalInputSize(Exception):
    """
    The input data amount cannot match the required size,Pls increase input data amount or increase the max_limit
    """
    pass

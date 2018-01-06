import pandas as pd
from Frcwp import Frcwp

path = 'data/data_all.csv'
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
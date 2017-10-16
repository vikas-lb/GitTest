import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import gc
import os
os.chdir("C:\Users\Vikas19\Documents\kaggle")
os.getcwd()

print('loading files...')
train = pd.read_csv(os.getcwd()+ '\\input\\train.csv', na_values=-1)
test = pd.read_csv(os.getcwd()+ '\\input\\test.csv', na_values=-1)
col_to_drop = train.columns[train.columns.str.startswith('ps_calc_')]
train = train.drop(col_to_drop, axis=1)  
test = test.drop(col_to_drop, axis=1)  

for c in train.select_dtypes(include=['float64']).columns:
    train[c]=train[c].astype(np.float32)
    test[c]=test[c].astype(np.float32)
for c in train.select_dtypes(include=['int64']).columns[2:]:
    train[c]=train[c].astype(np.int8)
    test[c]=test[c].astype(np.int8)    

print(train.shape, test.shape)

# custom objective function (similar to auc)

def gini(y, pred):
    g = np.asarray(np.c_[y, pred, np.arange(len(y)) ], dtype=np.float)
    g = g[np.lexsort((g[:,2], -1*g[:,1]))]
    gs = g[:,0].cumsum().sum() / g[:,0].sum()
    gs -= (len(y) + 1) / 2.
    return gs / len(y)

def gini_xgb(pred, y):
    y = y.get_label()
    return 'gini', gini(y, pred) / gini(y, y)

def gini_lgb(preds, dtrain):
    y = list(dtrain.get_label())
    score = gini(y, preds) / gini(y, y)
    return 'gini', score, True
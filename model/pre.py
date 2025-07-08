#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import lightgbm as lgb
import joblib  # 用于保存标准化器

from sklearn import tree,svm,neighbors,ensemble,gaussian_process
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.tree import ExtraTreeRegressor

def load_data(data_name, fp_name):
    data = pd.read_csv(data_name+'.csv', header=0) 
    fingerprint = pd.read_csv(fp_name+'.csv', header=None)
    y = data['gap'].values
    X = fingerprint.values
    X = np.nan_to_num(X)
    st = StandardScaler()
    X = st.fit_transform(X)
    train_data = lgb.Dataset(X,y)
    
    params = {'nthread': 9,
              'max_depth': 8,
              'min_child_samples': 11,
              'learning_rate': 0.1671180461605235,
              'num_leaves': 128,
              'subsample_for_bin': 1000,
              'reg_alpha': 0.1060904717573761,
              'reg_lambda': 0.21781669819047828,
              'bagging_seed': 100,
              'colsample_bytree': 0.8352937073744743,
              'is_unbalance':True,
              'verbose':-1}
    
    params['boosting_type']='dart'
    params['subsample'] = 0.8947714597497915
    params['metric'] = ['rmse']
    params['feature_pre_filter'] = False

    model = lgb.train(params, train_data, num_boost_round=1000)
    
    # ===== 保存模型和标准化器 =====
    model.save_model('lgbm_model.txt')  # 保存LightGBM模型
    joblib.dump(st, 'std_scaler.bin', protocol=4)     # 保存标准化器
    # =============================
    
    return model, st  # 返回模型和标准化器以备后续使用

if __name__ == '__main__':
    data_name1 = 'gap'
    fp_name1 = 'de'
    model, scaler = load_data(data_name1, fp_name1)  # 接收返回的模型和标准化器

		
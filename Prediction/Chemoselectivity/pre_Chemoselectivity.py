#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import lightgbm as lgb
 
from sklearn.preprocessing import StandardScaler
from rdkit import RDLogger
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from pathlib import Path

current_dir = Path(__file__).parent
sufex_dir = current_dir.parent.parent
data_dir = sufex_dir / "Data"


def get_des(smi):
    if not pd.isna(smi):
        try:
            mol = Chem.MolFromSmiles(smi)
            mol = AllChem.AddHs(mol)
            calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
            ds = np.asarray(calc.CalcDescriptors(mol))
            arr = AllChem.GetMACCSKeysFingerprint(mol)
            arr = np.asarray(arr)
            return np.append(arr, ds)
        except:
            print(smi)
            return np.zeros(377)
    else:
        return np.zeros(377)

def df2list(df):
    return pd.DataFrame(df.apply(get_des).values.tolist())

def load_data(data_name1, fp_name1, fp_name2):
	data1 = pd.read_excel(data_name1, header=0) 
	fingerprint1 = pd.read_csv(fp_name1, header=None)
	y1 = data1['yield'].values
	X1 = fingerprint1.values
	X1 = np.nan_to_num(X1)
	st = StandardScaler()
	X1 = st.fit_transform(X1)
	X2 = fp_name2.values
	X2 = np.nan_to_num(X2)
	X2 = st.transform(X2)

	train_data = lgb.Dataset(X1,y1)

	params = {'nthread': 8,  # 进程数
              'max_depth': 8,  # 最大深度
              'min_child_samples': 12,  # 树的数量
              'learning_rate': 0.2033490071950328,  # 学习率
              'num_leaves': 256,  # 终点节点最小样本占比的和
              'subsample_for_bin': 1000,  # 样本列采样
              'reg_alpha': 0.4458830183950115,  # L1 正则化
              'reg_lambda': 0.8332690107043002,  # L2 正则化
              'bagging_seed': 100,# 随机种子,light中默认为100
              'colsample_bytree': 0.7891360067061451,
              'is_unbalance':False,
              'verbose':-1
               }
	params['boosting_type']='dart'
	params['subsample'] = 0.9020560157043988
	params['metric'] = ['rmse']
	params['feature_pre_filter'] = False

	model = lgb.train(params,train_data,num_boost_round=1000)
	prediction = model.predict(X2, num_iteration=model.best_iteration)
	return prediction


def get_favored_product(pro1_yield, pro2_yield):

    if pro1_yield > pro2_yield:
        return 'pro1'
    elif pro1_yield < pro2_yield:
        return 'pro2'


def format_yield_description(row):
    pro1_yield_pct = f"{row['pro1_yield']/100:.1%}"
    pro2_yield_pct = f"{row['pro2_yield']/100:.1%}"
    
    if row['favored_product'] == 'pro1':
        favored_text = "Product1 is more favourable"
    elif row['favored_product'] == 'pro2':
        favored_text = "Product2 is more favourable"
  
    return f"The yield of Product1 is {pro1_yield_pct}, The yield of Product2 is {pro2_yield_pct}, {favored_text}"


RDLogger.DisableLog('rdApp.error')    
RDLogger.DisableLog('rdApp.warning') 
df = pd.read_excel('pre_data.xlsx')
des_base = df2list(df['Base'])
des_base_eq=df['equiv(Base)']
des_Reactant1 = df2list(df['Reactant1'])
des_Reactant1_eq = df['equiv(Reactant1)']
des_Reactant2 = df2list(df['Reactant2'])
des_pro1 = df2list(df['Product1'])
des_pro2 = df2list(df['Product2'])
des_diff1 = des_pro1 - des_Reactant1 - des_Reactant2
des_diff2 = des_pro2 - des_Reactant1 - des_Reactant2
des_solvent1 = df2list(df['solvent'])
des_ET=df['E']
des_Addictive1 = df2list(df['Additive1'])
des_Addictive1_eq = df['equiv(Additive1)']
des_Addictive2 = df2list(df['Additive2'])
des_Addictive2_eq = df['equiv(Additive2)']
des_time = df['time']
des_T = df['T']
des_Al = df['MS']
des_MW = df['MW']
des_com1 = pd.concat([des_base, des_base_eq, des_diff1, des_Reactant1_eq, des_solvent1, des_ET, des_Addictive1,
                         des_Addictive1_eq, des_Addictive2, des_Addictive2_eq, des_MW, des_time, des_T, des_Al], axis=1, ignore_index=True)
des_com1.reset_index(drop=True, inplace=True)

des_com2 = pd.concat([des_base, des_base_eq, des_diff2, des_Reactant1_eq, des_solvent1, des_ET, des_Addictive1,
                         des_Addictive1_eq, des_Addictive2, des_Addictive2_eq, des_MW, des_time, des_T, des_Al], axis=1, ignore_index=True)
des_com2.reset_index(drop=True, inplace=True)

data_name1 = data_dir / "dataset.xlsx"
fp_name1 = data_dir / "dataset_Rdkit_MACCS_DielectricConstant.csv"
pre1  = load_data(data_name1, fp_name1, des_com1)
pre2 = load_data(data_name1, fp_name1, des_com2)
df['pro1_yield'] = pre1
df['pro2_yield'] = pre2
df['favored_product'] = df.apply(
    lambda row: get_favored_product(row['pro1_yield'], row['pro2_yield']), 
    axis=1
)


df['chemoselectivity'] = df.apply(format_yield_description, axis=1)
df = df.drop(columns=['pro1_yield', 'pro2_yield', 'favored_product'])

output_file = 'pre_data_chemoselectivity.xlsx'
df.to_excel(output_file, index=False)


		


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

    params = {'nthread': 8,  
              'max_depth': 8,  
              'min_child_samples': 12,  
              'learning_rate': 0.2033490071950328,  
              'num_leaves': 256,  
              'subsample_for_bin': 1000,  
              'reg_alpha': 0.4458830183950115,  
              'reg_lambda': 0.8332690107043002,  
              'bagging_seed': 100,
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

  
RDLogger.DisableLog('rdApp.error')    
RDLogger.DisableLog('rdApp.warning') 
df = pd.read_excel('pre_data.xlsx')
des_base = df2list(df['Base'])
des_base_eq=df['equiv(Base)']
des_Reactant1 = df2list(df['Reactant1'])
des_Reactant1_eq = df['equiv(Reactant1)']
des_Reactant2 = df2list(df['Reactant2'])
des_pro = df2list(df['Product'])
des_diff = des_pro - des_Reactant1 - des_Reactant2
des_solvent1 = df2list(df['solvent'])
des_E=df['DielectricConstant']
des_Addictive1 = df2list(df['Additive1'])
des_Addictive1_eq = df['equiv(Additive1)']
des_Addictive2 = df2list(df['Additive2'])
des_Addictive2_eq = df['equiv(Additive2)']
des_time = df['time']
des_T = df['T']
des_MS = df['MS']
des_MWI = df['MWI']
des_com = pd.concat([des_base, des_base_eq, des_diff, des_Reactant1_eq, des_solvent1, des_E, des_Addictive1,
                         des_Addictive1_eq, des_Addictive2, des_Addictive2_eq, des_MWI, des_time, des_T, des_MS], axis=1, ignore_index=True)
des_com.reset_index(drop=True, inplace=True)

data_name1 = data_dir / "dataset.xlsx"
fp_name1 = data_dir / "dataset_Rdkit_MACCS_DielectricConstant.csv"
pre  = load_data(data_name1, fp_name1, des_com)
pre = pd.DataFrame(pre)
pre.to_csv('pre_data_yield.csv', index=False, header=False)


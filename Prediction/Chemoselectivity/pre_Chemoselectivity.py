

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


def get_favored_product(pro1_yield, pro2_yield):

    if pro1_yield > pro2_yield:
        return 'pro1'
    elif pro1_yield < pro2_yield:
        return 'pro2'


def format_yield_description(row):
    pro1_yield_pct = f"{row['pro1_yield']/100:.1%}"
    pro2_yield_pct = f"{row['pro2_yield']/100:.1%}"
    
    if row['favored_product'] == 'pro1':
        return f"Product1 is more favourable, the yield is {pro1_yield_pct}"
    elif row['favored_product'] == 'pro2':
        return f"Product2 is more favourable, the yield is {pro2_yield_pct}"
  


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
des_E=df['DielectricConstant']
des_Addictive1 = df2list(df['Additive1'])
des_Addictive1_eq = df['equiv(Additive1)']
des_Addictive2 = df2list(df['Additive2'])
des_Addictive2_eq = df['equiv(Additive2)']
des_time = df['time']
des_T = df['T']
des_MS = df['MS']
des_MWI = df['MWI']
des_com1 = pd.concat([des_base, des_base_eq, des_diff1, des_Reactant1_eq, des_solvent1, des_E, des_Addictive1,
                         des_Addictive1_eq, des_Addictive2, des_Addictive2_eq, des_MWI, des_time, des_T, des_MS], axis=1, ignore_index=True)
des_com1.reset_index(drop=True, inplace=True)

des_com2 = pd.concat([des_base, des_base_eq, des_diff2, des_Reactant1_eq, des_solvent1, des_E, des_Addictive1,
                         des_Addictive1_eq, des_Addictive2, des_Addictive2_eq, des_MWI, des_time, des_T, des_MS], axis=1, ignore_index=True)
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


df['Pred_Pro'] = df.apply(format_yield_description, axis=1)
df = df.drop(columns=['pro1_yield', 'pro2_yield', 'favored_product'])

output_file = 'pre_data_chemoselectivity.xlsx'
df.to_excel(output_file, index=False)


		

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
import pandas as pd
import numpy as np
from datetime import datetime
from rdkit import RDLogger
from pathlib import Path

current_dir = Path(__file__).parent
sufex_dir = current_dir.parent
data_dir = sufex_dir / "Data"

start = datetime.now()


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

RDLogger.DisableLog('rdApp.error')    
RDLogger.DisableLog('rdApp.warning')

filename = data_dir / "dataset.xlsx" 
df = pd.read_excel(filename)
des_base = df2list(df['Base'])
des_base_eq=df['equiv(Base)']
des_Reactant1 = df2list(df['Reactant1'])
des_Reactant1_eq = df['equiv(Reactant1)']
des_Reactant2 = df2list(df['Reactant2'])
des_pro = df2list(df['Product'])
des_diff = des_pro - des_Reactant1 - des_Reactant2
des_solvent1 = df2list(df['solvent'])
des_Addictive1 = df2list(df['Additive1'])
des_Addictive1_eq = df['equiv(Additive1)']
des_Addictive2 = df2list(df['Additive2'])
des_Addictive2_eq = df['equiv(Additive2)']
des_time = df['time']
des_T = df['T']
des_MS = df['MS']
des_MWI = df['MWI']
des_com = pd.concat([des_base, des_base_eq, des_diff, des_Reactant1_eq, des_solvent1, des_Addictive1,
                         des_Addictive1_eq, des_Addictive2, des_Addictive2_eq, des_MWI, des_time, des_T, des_MS], axis=1, ignore_index=True)
des_com.reset_index(drop=True, inplace=True)
des_com.to_csv('dataset_Rdkit_MACCS.csv', index=False, header=False)
end = datetime.now()
print(des_com.shape)
print('Total Time is %ss! ' %(end-start))

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# add multiprocessing 2021-06-13

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys
from rdkit.ML.Descriptors import MoleculeDescriptors
# import multiprocessing as mp
import pandas as pd
import numpy as np
from datetime import datetime
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

#读入数据
if __name__ == '__main__':
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.error')    
    RDLogger.DisableLog('rdApp.warning') 
    df = pd.read_csv('gap.csv')
    des_com = df2list(df['SMILES'])
    des_com.reset_index(drop=True, inplace=True)
    des_com.to_csv('de.csv', index=False, header=False)
    end = datetime.now()
    print(des_com.shape)
    print('Total Time is %ss! ' %(end-start))


import pandas as pd
import numpy as np
import xgboost as xgb
 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error
from pathlib import Path

current_dir = Path(__file__).parent
sufex_dir = current_dir.parent
data_dir = sufex_dir / "Data"



def load_data(data_name, fp_name):
	data = pd.read_excel(data_name, header=0) 
	fingerprint = pd.read_csv(fp_name, header=None)
	y = data['yield'].values
	X = fingerprint.values
	X = np.nan_to_num(X)
	st = StandardScaler()
	X = st.fit_transform(X)

	X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size=0.8, random_state=20)

	Train_data = xgb.DMatrix(X_train,Y_train)
	Test_data = xgb.DMatrix(X_test,Y_test)
	
	kf = KFold(n_splits=5, random_state=10, shuffle=True)
	r2 = []
	for train_index,val_index in kf.split(X_train,Y_train):
		x_train,x_val = X[train_index],X[val_index]
		y_train,y_val = y[train_index],y[val_index]
		train_data = xgb.DMatrix(x_train,y_train)
		test_data = xgb.DMatrix(x_val,y_val) 
		space = {"eta": 0.05
                     ,"booster": 'gbtree'
                     ,"colsample_bytree":0.3
                     ,"colsample_bynode":0.4
                     ,"gamma":0.001
                     ,"lambda":1.2
                     ,"min_child_weight":4.0
                     ,"max_depth":14
                     ,"subsample":0.9
                     ,"objective":'reg:squarederror'
					 ,"nthread":12
                    }	
		model = xgb.train(space,train_data, num_boost_round=600, evals=[(test_data,'eval')], early_stopping_rounds=100)
		prediction = model.predict(test_data)
		prediction = np.clip(prediction, 0, 100)
		R2=r2_score(y_val, prediction)
		r2.append(R2)
	print(r2)
	r2_mean=np.mean(r2)
	print(r2_mean)

	model = xgb.train(space,Train_data, num_boost_round=600, evals=[(Test_data,'eval')], early_stopping_rounds=100)
	prediction1 = model.predict(Test_data)
	prediction1 = np.clip(prediction1, 0, 100)
	RMSE=np.sqrt(mean_squared_error(Y_test, prediction1))
	R2_test=r2_score(Y_test, prediction1)
	print('R2_test:',R2_test)
	print('RMSE:',RMSE)
	with open('train.txt', 'w') as f:
		f.write(str(r2)+'\n')
		f.write(str(r2_mean)+'\n')
		f.write(str(R2_test)+'\n')
		f.write(str(RMSE))

	
if __name__ == '__main__':
	# file name
	data_name = data_dir / "dataset.xlsx"
	fp_name = data_dir / "dataset_Rdkit_MACCS.csv"
	load_data(data_name, fp_name)

		
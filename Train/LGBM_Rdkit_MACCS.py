
import pandas as pd
import numpy as np
import lightgbm as lgb

 
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
	Train_data = lgb.Dataset(X_train,Y_train)
	Test_data = lgb.Dataset(X_test,Y_test)


	kf = KFold(n_splits=5, random_state=10, shuffle=True)
	r2 = []

	for train_index,val_index in kf.split(X_train, Y_train):
		x_train,x_val = X[train_index],X[val_index]
		y_train,y_val = y[train_index],y[val_index]
		train_data = lgb.Dataset(x_train,y_train)
		test_data = lgb.Dataset(x_val,y_val) 
		params = {'nthread': 8, 
              'max_depth': 11, 
              'min_child_samples': 28, 
              'learning_rate': 0.25823859310798225, 
              'num_leaves': 2048, 
              'subsample_for_bin': 1000, 
              'reg_alpha': 0.8300219846523877,  
              'reg_lambda': 0.3522383690153793,  
              'bagging_seed': 100,
              'colsample_bytree': 0.28622486781725637,
              'is_unbalance':False,
              'verbose':-1
               }
		params['boosting_type']='dart'
		params['subsample'] = 0.6197217450424286
		params['metric'] = ['rmse']
		params['feature_pre_filter'] = False
		model = lgb.train(params,train_data,num_boost_round=1000)
		prediction = model.predict(x_val, num_iteration=model.best_iteration)
		prediction = np.clip(prediction, 0, 100)
		R2=r2_score(y_val, prediction)
		r2.append(R2)
	print(r2)
	r2_mean=np.mean(r2)
	print(r2_mean)

	model = lgb.train(params,Train_data,num_boost_round=1000)
	prediction1 = model.predict(X_test, num_iteration=model.best_iteration)
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

		
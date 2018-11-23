
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train=pd.read_csv('train.csv')
x=train.iloc[:,:80].values
y=train.iloc[:,80].values

test=pd.read_csv('test.csv')
x_test=test.iloc[:,79].values


indx=test['Id'].copy()
train=train.drop('Id',1)
test=test.drop('Id',1)

#SEE ALL THE DATA TYPES PRESENT
train.dtypes


#FOR TRAIN SET
#SPLITTING CATEGORICAL AND NUMERIC DATA
cat_fea=train.dtypes.loc[train.dtypes=='object'].index
int_fea=train.dtypes.loc[train.dtypes=='int64'].index
float_fea=train.dtypes.loc[train.dtypes=='float'].index


train[cat_fea].apply(lambda x: len(x.unique()))
train[int_fea].apply(lambda x: len(x.unique()))


#COUNTING THE NUMBER OF MISSING VAUES IN BOTH CATEGORICAL AND NUMERIC DATA
train[cat_fea].apply(lambda x: x.isnull().sum())
train[int_fea].apply(lambda x: x.isnull().sum())
train[float_fea].apply(lambda x: len(x.unique()))


nulls=train.isnull().sum()


#TREATING MISSING DATA FOR CATEGORICAL VAR
for var in cat_fea:
    r=train[var].mode()
    r[0]
    train[var].fillna(r[0],inplace=True)


#TREATING MISSING DATA FOR NUMERIC VAR
for var in int_fea:
    r=train[var].mode()
    r[0]
    train[var].fillna(r[0],inplace=True)


#TREATING MISSING DATA FOR FLOAT VAR
for var in float_fea:
    r=train[var].mode()
    r[0]
    train[var].fillna(r[0],inplace=True)




#LABEL ENCODING THE CATEGORICAL VALUES
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le=LabelEncoder()
for var in cat_fea:
    train[var]=le.fit_transform(train[var])
    

#FOR TEST SET
    
#SPLITTING CATEGORICAL AND NUMERIC DATA
cat_fea=test.dtypes.loc[test.dtypes=='object'].index
int_fea=test.dtypes.loc[test.dtypes=='int64'].index
float_fea=test.dtypes.loc[test.dtypes=='float'].index

test[cat_fea].apply(lambda x: len(x.unique()))
test[int_fea].apply(lambda x: len(x.unique()))
test[float_fea].apply(lambda x: len(x.unique()))

#COUNTING THE NUMBER OF MISSING VAUES IN BOTH CATEGORICAL AND NUMERIC DATA
test[cat_fea].apply(lambda x: x.isnull().sum())
test[int_fea].apply(lambda x: x.isnull().sum())
test[float_fea].apply(lambda x: len(x.unique()))


nulls=test.isnull().sum()


#TREATING MISSING DATA FOR CATEGORICAL VAR
for var in cat_fea:
    r=test[var].mode()
    r[0]
    test[var].fillna(r[0],inplace=True)


#TREATING MISSING DATA FOR NUMERIC VAR
for var in int_fea:
    r=test[var].mode()
    r[0]
    test[var].fillna(r[0],inplace=True)

#TREATING MISSING DATA FOR FLOAT VAR
for var in float_fea:
    r=test[var].mode()
    r[0]
    test[var].fillna(r[0],inplace=True)


#LABEL ENCODING THE CATEGORICAL VALUES
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le=LabelEncoder()

for var in cat_fea:
    test[var]=le.fit_transform(test[var]) 


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train = sc.fit_transform(train)
test = sc.fit_transform(test)

	
#SPLIT DATASETS
x=train[:,0:80]
y=train[:,79]
x_test=test[:,:79]


#apply kernel pca after data perprocessing steps..
from sklearn.decomposition import KernelPCA as KPC
kpca = KPC(n_components=2,kernel='rbf' )
x_train=kpca.fit_transform(x_train)
x_test=kpca.transform(x_test)


# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(C=10000,gamma=1,kernel = 'rbf')
regressor.fit(x, y)
y_pre=regressor.predict(x)

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)
from sklearn.model_selection import cross_val_score as cvs
acc=[]
acc=cvs(estimator=lin_reg,X=x_train,y=y_train,cv=10)
#3 WAYS TO FIND MEAN
np.mean(acc)
y_pred_normal=lin_reg.predict(x_test)
from sklearn.model_selection import GridSearchCV
#dictionaries are initialised by "{}" in python
parameters=[{'C':[10,100 ,1000,2000,10000], 'gamma': [1,0,2,3,10,100]}]#see help of these functions before initialising values
#here we specify different parametrs combo and the grid search chooses the optimal ones.
grid_search=GridSearchCV(estimator=regressor, param_grid=parameters ,cv=10)
grid_search.fit(x,y)
#below are inbuilt 
best_est=grid_search.best_estimator_
best_score=grid_search.best_score_
bes_para=grid_search.best_params_
y_pred_new_normal=grid_search.predict(test)



#KFOLD CROSS VAL
from sklearn.model_selection import cross_val_score as cvs
acc=[]
acc=cvs(estimator=regressor,X=x_train,y=y_train,cv=10)
#3 WAYS TO FIND MEAN
np.mean(acc)

# Predicting a new result
y_pred_normal = regressor.predict(x_test)


#APPLY grid search to find the best model(linear or non linear ) and to find the best hyper-parameters
#like C, LAMBDA etc
from sklearn.model_selection import GridSearchCV
#dictionaries are initialised by "{}" in python
parameters=[{'C': [1,10,100,1000], 'kernel':['linear']} ,
            {'C':[1,10,100,1000] , 'kernel':['rbf'] ,'gamma': [0.5,0.3,0.4,0.6,0.7,0.56,0.9,1]}]#see help of these functions before initialising values
#see help of these functions before initialising values
#here we specify different parametrs combo and the grid search chooses the optimal ones.
grid_search=GridSearchCV(estimator=regressor, param_grid=parameters ,cv=10)
grid_search.fit(x_train,y_train)
#below are inbuilt 
best_est=grid_search.best_estimator_
best_score=grid_search.best_score_
bes_para=grid_search.best_params_
y_pred_new_normal=grid_search.predict(x_test)

#STEP 2----trying prediction with xgboost....THALA

#FITTING THE XGBOOST TO TRAIN
import xgboost
from xgboost import XGBRegressor  
regressor_xg=XGBRegressor()
regressor_xg.fit(x,y)

#apply kfold cv
from sklearn.model_selection import cross_val_score as cvs
accuracies=[]
accuracies=cvs(estimator=regressor_xg,X=x,y=y,cv=10)
avg=accuracies.mean()

x_test=test.values[:,79]
#predictions
y_pred_xg=regressor_xg.predict(x_test)

from sklearn.metrics import mean_squared_error as mse
from math import sqrt
acc=sqrt(mse(y_test,y_pred_xg))

submission=pd.DataFrame({'Id': indx,'SalePrice':y_pred_xg})
submission2=pd.DataFrame({'Id': indx,'SalePrice':y_pred_new_normal})

submission.to_csv('ans_gcv.csv',index=False)



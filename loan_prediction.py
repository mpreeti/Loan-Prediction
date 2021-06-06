# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
#%%Loading training n testing DS
train_data=pd.read_csv(r'C:\Users\hp\Desktop\loan_prediction_train.csv',header=0)
test_data=pd.read_csv(r'C:\Users\hp\Desktop\loan_prediction_test.csv',header=0)
print(train_data.shape)
train_data.head()
#%%
#finding missing values
train_data.isnull().sum()
train_data.describe(include="all")
#%%filling missing val

colname1=['Gender','Married','Dependents','Self_Employed','Loan_Amount_Term']
for x in colname1[:]:
    train_data[x].fillna(train_data[x].mode()[0],inplace=True) 
train_data.isnull().sum()

#%%

train_data["LoanAmount"].fillna(train_data["LoanAmount"].mean(),inplace=True)
print(train_data.isnull().sum())
#%%

train_data['Credit_History'].fillna(value=0,inplace=True)
print(train_data.isnull().sum())
#%%converting cat to num
colname=['Gender','Married','Education','Dependents','Self_Employed','Property_Area','Loan_Status']
colname
#For preprocessing the data
from sklearn import preprocessing
le={}
for x in colname:
    le[x]=preprocessing.LabelEncoder()
    
for x in colname:
    train_data[x]=le[x].fit_transform(train_data.__getattr__(x))
#converted Loan status as Y-->1 and N-->0
train_data.head()
#%%Preprcessing test ds
print(test_data.shape)
test_data.head()

test_data.isnull().sum()
test_data.describe(include="all")

colname1=['Gender','Dependents','Self_Employed','Loan_Amount_Term']
for x in colname1[:]:
    test_data[x].fillna(test_data[x].mode()[0],inplace=True) 
test_data.isnull().sum()


test_data["LoanAmount"].fillna(test_data["LoanAmount"].mean(),inplace=True)
print(test_data.isnull().sum())


test_data['Credit_History'].fillna(value=0,inplace=True)
print(test_data.isnull().sum())

colname=['Gender','Married','Education','Dependents','Self_Employed','Property_Area']
colname

from sklearn import preprocessing
le={}
for x in colname:
    le[x]=preprocessing.LabelEncoder()
    
for x in colname:
    test_data[x]=le[x].fit_transform(test_data.__getattr__(x))
#Y-->1 N-->0
test_data.head()
#%%
X_train=train_data.values[:,1:-1]
Y_train=train_data.values[:,-1]
Y_train=Y_train.astype(int)
Y_train.dtype

X_test=test_data.values[:,1:] 

#%%scaling
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler() 
scaler.fit(X_train)
X_train=scaler.transform(X_train)
print(X_train)

scaler.fit(X_test)
X_test=scaler.transform(X_test)
print(X_test)

#%%SVM modelling
from sklearn import svm
svc_model=svm.SVC(kernel='rbf',C=1.0,gamma=0.1)

svc_model.fit(X_train,Y_train)
Y_pred=svc_model.predict(X_test)
print(list(Y_pred))
Y_pred_col=list(Y_pred)

#%%appending predicted loan status in test data file 
test_data=pd.read_csv(r'C:\Users\hp\Desktop\loan_prediction_test.csv',header=0)
test_data["Y_predictions"]=Y_pred_col
test_data.head()

#%%
submission = pd.DataFrame({'Loan_ID': test_data['Loan_ID'],'Loan_Status': Y_pred_col})

submission['Loan_Status'] =np.where(submission.Loan_Status ==1,'Y','N')
submission.to_csv('submission.csv',index=False)







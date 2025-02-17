#%% Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import torch


#%% Importing data
train_transaction = pd.read_csv('ieee-fraud-detection/train_transaction.csv')


#%% Exploring the transaction data
print(train_transaction.head())
print(train_transaction.info())
print(train_transaction.describe())


# %% Exploring the identity data
train_identity = pd.read_csv('ieee-fraud-detection/train_identity.csv')
print(train_identity.head())
print(train_identity.info())
print(train_identity.describe())

# %%
train_identity
# %% Fraud counts
train_transaction['isFraud'].value_counts()
# %% Duplica
train_transaction.duplicated().value_counts()
# %% transaction NA
train_transaction.isna().sum().sort_values(ascending=False)
# %% dist columns
for col in train_transaction.columns:
    if col.startswith('d'):
        print(col)
# %% dist columns
#train_transaction['dist2'].value_counts()
#train_transaction[['dist1', 'dist2']]
train_transaction[['dist1', 'dist2']].describe()

# %%
numerical_cols_tran = train_transaction.select_dtypes(include=['int64', 'float64']).columns
isFraud_col = train_transaction[numerical_cols_tran[1]]
num_cols_less_na_tran = numerical_cols_tran[train_transaction[numerical_cols_tran].isna().sum() < 0.75 * len(train_transaction)].drop('isFraud')


#%% SimpleImputer for train_transaction
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
train_transaction1 = imputer.fit_transform(train_transaction[num_cols_less_na_tran])
train_transaction1


#%% Split columns into X and y
y_train = isFraud_col.values
X_train = train_transaction1

#%%
y_train

# %% Standardizing the data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
X_scaled_train = scaler.transform(X_train)
X_scaled_train


# %% Initial model - Logistic Regression
from sklearn.linear_model import LogisticRegression

model_tran = LogisticRegression()
#model.fit(X_train, y_train)


#%% K-fold cross validation
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled_train, y_train)):
    X_train1, X_val1 = X_scaled_train[train_idx], X_scaled_train[val_idx]
    y_train1, y_val1 = y_train[train_idx], y_train[val_idx]
    
    model_tran.fit(X_train1, y_train1)
    y_pred = model_tran.predict(X_val1)
    print(f"Fold {fold + 1} Accuracy:", accuracy_score(y_val1, y_pred))


# %% Train full data with Logistic Regression
model_tran.fit(X_scaled_train, y_train)

# %% Import test data
test_transaction = pd.read_csv('ieee-fraud-detection/test_transaction.csv')
#test_identity = pd.read_csv('ieee-fraud-detection/test_identity.csv')

# %%
common_cols = num_cols_less_na_tran.intersection(test_transaction.columns)

test_transaction_subset = test_transaction[common_cols]

#%%
test_transaction_subset

#%% Impute missing values with SimpleImputer
X_imputed_test = imputer.transform(test_transaction_subset)
#%% Scaling test data
X_scaled_test = scaler.transform(X_imputed_test)
#%%
X_scaled_test


# %% Predicting test data
y_pred_test = model_tran.predict_proba(X_scaled_test)[:,1]
#[:,1]
round(y_pred_test, 2)

# %% Histogram of the predicted probabilities
plt.hist(y_pred_test[y_pred_test>0.5], bins=100)
plt.show()

#%% Appending Transaction ID and predicted probabilities
test_result_transaction = pd.DataFrame()
test_result_transaction['TransactionID'] = test_transaction['TransactionID'].astype(str)
test_result_transaction['isFraud'] = y_pred_test

#%%
test_result_transaction['isFraud'] = round(test_result_transaction['isFraud'], 1)

# %%
test_result_transaction.to_csv('test_result_transaction.csv', index=False)
# %%

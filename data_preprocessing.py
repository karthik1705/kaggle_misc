#%% Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
numerical_cols = train_transaction.select_dtypes(include=['int64', 'float64']).columns

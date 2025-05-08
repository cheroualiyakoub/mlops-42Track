#!/usr/bin/env python
# coding: utf-8

# In[1641]:


import pandas as pd
import seaborn as sns
import logging
import numpy as np


# In[1642]:


import matplotlib.pyplot as plt
import math


# In[1643]:


from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
import statsmodels.api as sm
from sklearn.feature_selection import RFE

from sklearn.metrics import root_mean_squared_error

from sklearn.preprocessing import MinMaxScaler



# In[1644]:


# pipline funtion

def read_data(filename):
    try:
        df = pd.read_csv(filename)
        if df.empty:
            raise ValueError("🚨 Data loaded but is empty.")
        return df
    except FileNotFoundError:
        raise FileNotFoundError("❌ Data file not found at specified path.")
    except Exception as e:
        raise RuntimeError(f"❌ Unexpected error while loading data: {e}")
    


# In[1645]:


def print_infos(df):
    print("Data info : \n")
    display(df.info())

    print ("\n\nData describtion: \n")
    display(df.describe())


# In[1646]:


def check_null(df):
    print(df.isnull().sum().sort_values(ascending=False))
    


# In[1647]:


# pipline funtion

def intial_clean_data(df):
    cols_to_drop = ['instant', 'dteday' , 'registered', 'casual']
    df = df.drop(columns = cols_to_drop)
    return df


# In[1648]:


def dispayl_histplot(df):
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    
    n_cols = 4  
    n_rows = math.ceil(len(numeric_columns) / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    
    axes = axes.flatten()
    
    for i, column in enumerate(numeric_columns):
        sns.histplot(df[column], kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {column}')
        
    for i in range(len(numeric_columns), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


# In[1649]:


def display_barplot(df, column):
    plt.figure(figsize = (10,4))
    plt.subplot(1,2,1)
    sns.barplot(x=column,y='cnt',data=df)


# In[1650]:


def display_violinplot(df):
    plt.figure(figsize=(20, 15))
    
    # List of feature columns to compare with cnt
    columns_to_plot = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 
                       'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
    
    # Loop and plot each in a subplot
    for idx, col in enumerate(columns_to_plot):
        plt.subplot(4, 3, idx + 1)
        if df[col].nunique() < 10:
            # Categorical - use boxplot
            sns.violinplot(x=col, y='cnt', data=df)
        else:
            # Continuous - use scatterplot
            sns.scatterplot(x=col, y='cnt', data=df, alpha=0.5)
        plt.title(f'{col} vs cnt')
    
    plt.tight_layout()
    plt.show()


# In[1651]:


def display_barplot_grid(df):
    display_barplot(df, 'yr')
    categorical_columns = df.select_dtypes(include=['category']).columns.tolist()
    n_cols = 2
    n_rows = len(categorical_columns)

    plt.figure(figsize=(14, 6 * n_rows))

    plot_index = 1
    print(categorical_columns)
    for col in categorical_columns:
        display_barplot(df, col)
    plt.tight_layout()
    plt.show()


# In[1652]:


def display_corelation(df):
    corr_matrix = df.corr(numeric_only=True)
    plt.figure(figsize=(25, 20))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Between Features')
    plt.tight_layout()
    plt.show()


# In[1653]:


# pipline funtion

def convert_to_category_type(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype('category')
        else:
            raise KeyError(f"Column '{col}' not found in DataFrame.")
    return df
    


# In[1654]:


# pipline funtion

def convert_dummies(df):
    df = pd.get_dummies(df)
    return df
    


# In[1655]:


# pipline funtion

def get_numeric_columns(df):
    return df.select_dtypes(include=['number']).columns.tolist()


# In[1656]:


# pipeline funtion

def scale_num_data(df):
    scaler = MinMaxScaler()
    num_vars = get_numeric_columns(df)
    df[num_vars] = scaler.fit_transform(df[num_vars])
    return scaler


# In[1657]:


def scale_features(X):
    feature_scaler = MinMaxScaler()  
    X_scaled = feature_scaler.fit_transform(X) 
    return feature_scaler, X_scaled


# In[1658]:


def scale_target(y):
    target_scaler = MinMaxScaler()  
    y_scaled = target_scaler.fit_transform(y.values.reshape(-1, 1)) 
    return target_scaler, y_scaled


# In[1659]:


def unscale_features(scaler, X_scaled):
    X_unscaled = scaler.inverse_transform(X_scaled)
    return X_unscaled


# In[1660]:


def unscale_target(scaler, y_scaled):
    y_unscaled = scaler.inverse_transform(y_scaled)
    return y_unscaled


# In[1661]:


# pipleline funciton
from sklearn.model_selection import train_test_split

def data_split(df, target_column='cnt', test_size=0.15, val_size=0.15, random_state=42):
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # First, split off the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Calculate validation size relative to remaining data
    val_relative_size = val_size / (1 - test_size)

    # Split the remaining into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative_size, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

    


# In[1662]:


# pipleline cell

df = pd.read_csv("../data/hour.csv")
df.head(5)


# In[1663]:


print_infos(df)


# In[1664]:


# pipleline cell

check_null(df)


# In[1665]:


# pipleline cell

df_copy = df.copy()


# In[1666]:


df_copy


# In[1667]:


# pipleline cell

df_copy = intial_clean_data(df_copy)


# In[1668]:


df_copy


# In[1669]:


print_infos(df_copy)


# In[1670]:


dispayl_histplot(df_copy)


# In[ ]:





# In[1671]:


# pipleline cell

df_copy = convert_to_category_type(df_copy, ['weekday', 'weathersit', 'mnth', 'season'])


# In[1672]:


df_copy


# In[1673]:


print_infos(df_copy)


# In[ ]:





# In[1674]:


display_barplot_grid(df_copy)


# In[1675]:


sns.pairplot(data=df_copy,vars=['temp','atemp','hum','windspeed','cnt'])


# In[1676]:


display_violinplot(df_copy)


# In[1677]:


display_corelation(df_copy)


# In[1678]:


df_copy = df_copy.drop('temp', axis=1)


# In[1679]:


df_copy = convert_dummies(df_copy)


# In[1680]:


df_copy


# In[1681]:


print_infos(df_copy)


# In[1682]:


display_corelation(df_copy)


# In[1683]:


# pipleline cell

X_train, X_val, X_test, y_train, y_val, y_test = data_split(df_copy)


# In[1684]:


print(X_train.shape)
print(X_test.shape)
print(X_val.shape)


# In[1685]:


X_train


# In[1686]:


feature_scaler, X_train_scaled = scale_features(X_train)
target_scaler, y_train_scaled = scale_target(y_train)

X_val_scaled = feature_scaler.transform(X_val)
X_test_scaled = feature_scaler.transform(X_test)

y_test_scaled = scale_target(y_test)
y_val_scaled = scale_target(y_val)



# In[1687]:


y_val


# In[1688]:


y_val_scaled


# In[1689]:


X_val


# In[1690]:


# pipleline cell
lr_rfe = LinearRegression()
lr_rfe.fit(X_train_scaled, y_train_scaled)

rfe = RFE(estimator=lr_rfe, n_features_to_select=15)
rfe = rfe.fit(X_train_scaled, y_train_scaled)


# In[1691]:


# pipleline cell
X_train_rfe = pd.DataFrame(
    rfe.transform(X_train_scaled),
    columns=X_train.columns[rfe.support_],
    index=X_train.index
)


# In[1692]:


X_train_rfe


# In[1693]:


# pipleline cell

# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train_rfe.columns
vif['VIF'] = [variance_inflation_factor(X_train_rfe.values, i) for i in range(X_train_rfe.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[1694]:


df_copy


# In[1695]:


# Check for NaN or infinite values
if np.any(np.isnan(X_train_rfe)) or np.any(np.isinf(X_train_rfe)):
    print("There are NaN or infinite values in your data.")


# In[1696]:


constant_columns = [col for col in X_train.columns if X_train[col].var() == 0]
X_train_rfe


# In[1697]:


model = LinearRegression(fit_intercept=True)

# Add small noise to prevent perfect collinearity

model.fit(X_train_rfe, y_train_scaled)


# In[1698]:


X_test_rfe = pd.DataFrame(
    rfe.transform(X_val_scaled),
    columns=X_val.columns[rfe.support_],
    index=X_val.index
)



# In[1699]:


y_pred = model.predict(X_test_rfe)


# In[1700]:


nrmse = rmse / (y_test.max() - y_test.min())
print(f"Normalized RMSE: {nrmse:.2%}")


# In[1701]:


mean_y = y_test.mean()
relative_error = rmse / mean_y
print(f"Relative Error: {relative_error:.2%}")


# In[1702]:


print(y_pred.shape)
print(y_test.shape)


# In[1703]:


y_val_array


# In[1719]:


y_pred_original.min()
y_pred_scaled_clipped = np.clip(y_pred_original, 0, None)
y_pred_scaled_clipped.min()


# In[ ]:





# In[1720]:


from sklearn.metrics import mean_squared_log_error
import numpy as np

# Ensure your y_pred and y_test are numpy arrays
y_pred_array = np.array(y_pred).reshape(-1, 1)
y_val_array = np.array(y_val).reshape(-1, 1)

# Inverse transform
y_pred_original = unscale_target(target_scaler, y_pred_array)
y_pred_scaled_clipped = np.clip(y_pred_original, 0, None)
y_test_original = unscale_target(target_scaler, y_val_array)

# Compute RMSLE
rmsle = np.sqrt(mean_squared_log_error(y_test_original, y_pred_scaled_clipped))
print(f"RMSLE: {rmsle:.4f}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





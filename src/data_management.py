import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

def load_data(filename, skip_end_rows = 2):
    df = pd.read_csv(filename)
    df = df[:-skip_end_rows]
    return df

def preprocess_data(df):
    result_df = df.copy()
    result_df.set_index('Quarter', inplace=True)
    result_df.columns = result_df.columns.str.strip()
    for col in result_df.columns:
        if result_df[col].dtype == 'object':
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce')

    numeric_cols = result_df.select_dtypes(include=['float64', 'int64']).columns

    # Apply StandardScaler
    scaler = StandardScaler()
    result_df[numeric_cols] = scaler.fit_transform(result_df[numeric_cols])

    #Imputing missing values
    result_df.bfill(axis=0, inplace=True)
    result_df.ffill(axis=0, inplace=True)
    
    return result_df, scaler

def train_data_split(economic_data, n_future, n_past):
    test_data_position = 4 # GDP position
    trainX, trainY = [], []
    for i in range(n_past, len(economic_data) - n_future + 1):
        trainX.append(economic_data.iloc[i - n_past:i, 0 : economic_data.shape[1]].values)
        trainY.append(economic_data.iloc[i + n_future - 1:i + n_future, test_data_position].values)
    X, Y = np.array(trainX), np.array(trainY)
    
    train_size = int(0.9 * len(X))
    test_size = len(X) - train_size 

    print("train_size = ", train_size)
    print("test_size = ", test_size)
    X_train, y_train = X[:train_size], Y[:train_size]
    X_test, y_test = X[train_size:], Y[train_size:]

    print("train_size = ", X_train.shape)
    print("test_size = ", X_test.shape)

    return (X_train, y_train), (X_test, y_test)

def undo_scaling(y_test, scaler, column_index = 4):
    mean = scaler.mean_[column_index]
    std = scaler.scale_[column_index]

    y_test_unscaled = y_test * std + mean

    return y_test_unscaled
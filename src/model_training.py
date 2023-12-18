import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers.legacy import Adam, SGD
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GRU, Dropout
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout


def set_random_seed(seed_value = 42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)


def define_model_lstm(num_nodes, learning_rate, drop_out, activation, init_mode, optimizer, window, features):
    set_random_seed()
    optimizer_class = {'Adam': Adam, 'SGD': SGD}.get(optimizer)
    model = Sequential()
    model.add(LSTM(num_nodes, activation=activation, kernel_initializer=init_mode,
                input_shape=(window, features)))
    model.add(Dropout(drop_out))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer=optimizer_class(learning_rate=learning_rate),  metrics=[RootMeanSquaredError()])
    
    model.summary()
    return model


def define_model_gru(num_nodes, learning_rate, drop_out, activation, init_mode, optimizer, window, features):
    set_random_seed()
    optimizer_class = {'Adam': Adam, 'SGD': SGD}.get(optimizer)
    model = Sequential()
    model.add(GRU(num_nodes, activation=activation, kernel_initializer=init_mode, input_shape=(window, features)))
    model.add(Dropout(drop_out))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer=optimizer_class(learning_rate=learning_rate), metrics=[RootMeanSquaredError()])
    
    model.summary()
    return model


def define_model_cnn(num_nodes, learning_rate, drop_out, activation, init_mode, optimizer,kernel_size, window, features):
    set_random_seed()
    optimizer_class = {'Adam': Adam, 'SGD': SGD}.get(optimizer)
    model = Sequential()
    model.add(Conv1D(filters=num_nodes, kernel_size=kernel_size, activation=activation, kernel_initializer=init_mode,
                     input_shape=(window, features)))
    model.add(Dropout(drop_out))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(loss='mse', optimizer=optimizer_class(learning_rate=learning_rate), metrics=[RootMeanSquaredError()])
    
    model.summary()
    return model



def train_model(model, X_train, y_train, n_batch, epochs=100, is_early_stopping=False):
    print('X_train',X_train.shape)
    print('y_train',y_train.shape)
    callback = EarlyStopping(verbose=1, restore_best_weights=True, patience = 10)
    split_index = int(len(X_train) * 0.9)
    X_val, y_val = X_train[split_index:], y_train[split_index:]
    if is_early_stopping:
        trained_model = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=n_batch, callbacks=[callback], verbose=0)
    else:
        trained_model = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=n_batch, verbose=0)
    return trained_model



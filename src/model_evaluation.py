import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.optimizers.legacy import SGD
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import itertools
from math import sqrt
from model_training import define_model_lstm
from model_training import define_model_cnn
from model_training import define_model_gru
from model_training import train_model

class Info:
    def __init__(self):
        self.model = None
        self.params = None
        self.loss = None

def score_model(params, X_train, y_train , model_type):
    info = Info()
    num_nodes=params['num_nodes']
    learning_rate=params['learning_rate']
    drop_out=params['drop_out']
    activation=params['activation']
    init_mode=params['init_mode']
    optimizer=params['optimizer']
    batch_size=params['batch_size']
    kernel_size=params['kernel_size']
    window=16
    features=12
    if model_type=='LSTM':
        model = define_model_lstm(num_nodes, learning_rate, drop_out, activation, init_mode, optimizer, window,features)
    elif model_type=='CNN':
        model = define_model_cnn(num_nodes, learning_rate, drop_out, activation, init_mode, optimizer, kernel_size, window,features)
    else:
         model = define_model_gru(num_nodes, learning_rate, drop_out, activation, init_mode, optimizer, window,features)
    trained_model = train_model(model, X_train, y_train, batch_size)
    info.model = trained_model.model
    info.params = params
    info.loss = min(trained_model.history['val_root_mean_squared_error'])
    return info

def grid_search(param_grid,X_train,y_train,model_type):
    param_combinations = list(itertools.product(*param_grid.values()))
    param_dicts = [{key: value for key, value in zip(param_grid.keys(), combination)} for combination in param_combinations]
    list_of_trained_model=[]
    for params in param_dicts:
        info = score_model(params, X_train, y_train,model_type)
        list_of_trained_model.append(info)
    return list_of_trained_model


def pred_test(model, X_test, y_test):
    y_predict = model.predict(X_test).flatten()
    actual_data = y_test.flatten()
    test_results = pd.DataFrame({'Predictions': y_predict, 'Actuals': actual_data})
    rmse = sqrt(mean_squared_error(actual_data, y_predict))
    return test_results, rmse   

def actual_predict_plot(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df['Predictions'], label='Predictions On Test Data', marker='o')
    plt.plot(df['Actuals'], label='Actual Test Data', marker='x')
    plt.title('Predictions vs Actual')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)
    plt.show()

def war_no_war_plot(test_results_war_df, test_results_no_war_df, test_results_always_war_df):
    war_periods = {
        "Korean War": ("1950-06-25", "1953-07-27"),
        "Vietnam War": ("1965-03-08", "1975-04-30"),
        "Persian Gulf War": ("1990-08-02", "1991-02-28"),
        "War in Afghanistan": ("2001-10-07", "2021-08-30"),
        "Iraq War": ("2003-03-20", "2011-12-18"),
        "Russia Ukraine War": ("2022-02-24", "2023-09-30")
    }

    num_samples = len(test_results_war_df)
    dates = pd.date_range(start='2015-12-31', periods=num_samples, freq='Q')
    plt.figure(figsize=(12, 6))
    plt.plot(dates, test_results_war_df['Predictions'], label='GDP Predictions Due To US Involvement In Wars', color='red', marker='o')
    # plt.plot(dates, test_results_no_war_df['Predictions'], label='No War Predictions', color='blue', marker='x')
    # plt.plot(dates, test_results_always_war_df['Predictions'], label='Always War Predictions', color='orange', marker='D')
    start_date = pd.Timestamp('2015-12-31')
    for war, (start, end) in war_periods.items():
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        if end > start_date:
            plt.axvspan(max(start, start_date), end, color='lightcoral', alpha=0.3)

    # plt.title('Predicted Values: War vs No War vs Always War')
    plt.title('Prediction Of GDP During Wars')
    plt.xlabel('Date')
    plt.ylabel('Predicted Value')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45, ha='right')

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show() 

def plot_actual_war_test(df,test_results_war_df):

    war_periods = {
        "Korean War": ("1950-06-25", "1953-07-27"),
        "Vietnam War": ("1965-03-08", "1975-04-30"),
        "Persian Gulf War": ("1990-08-02", "1991-02-28"),
        "War in Afghanistan": ("2001-10-07", "2021-08-30"),
        "Iraq War": ("2003-03-20", "2011-12-18"),
        "Russia Ukraine War": ("2022-02-24", "2023-09-30")
    }

    num_samples = len(test_results_war_df)
    dates = pd.date_range(start='2015-12-31', periods=num_samples, freq='Q')
    plt.figure(figsize=(12, 6))
    plt.plot(dates, df['Actuals'], label='Actual GDP', marker='x',color='orange')
    plt.plot(dates, df['Predictions'], label='GDP Predictions Of Model Without War Variable', marker='o',color='green')
    plt.plot(dates, test_results_war_df['Predictions'], label='GDP Predictions of Model With War Variable', color='red', marker='o')
    start_date = pd.Timestamp('2015-12-31')
    for war, (start, end) in war_periods.items():
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        if end > start_date:
            plt.axvspan(max(start, start_date), end, color='lightcoral', alpha=0.3)

    plt.title('Prediction Of GDP On Test Data')
    plt.xlabel('Date')
    plt.ylabel('GDP Predicted Value In Billion Dollars')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45, ha='right')

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show() 


def plot_failure_analysis(train_results_war_df):
    war_periods = {
        "Korean War": ("1950-06-25", "1953-07-27"),
        "Vietnam War": ("1965-03-08", "1975-04-30"),
        "Persian Gulf War": ("1990-08-02", "1991-02-28"),
        "War in Afghanistan": ("2001-10-07", "2021-08-30"),
        "Iraq War": ("2003-03-20", "2011-12-18"),
        "Russia Ukraine War": ("2022-02-24", "2023-09-30")  # Assuming this is the current date range in your data
    }
    war_colors = {
        "Korean War": "lightblue",
        "Vietnam War": "lightgreen",
        "Persian Gulf War": "lightpink",
        "War in Afghanistan": "lightgrey",
        "Iraq War": "lightcoral",
        "Russia Ukraine War": "lightyellow"
    }
    start_date = pd.Timestamp('1952-03-31')
    end_date = pd.Timestamp('2023-06-30')
    dates = pd.date_range(start=start_date, end=end_date, freq='Q')
    min_length = min(len(dates), len(train_results_war_df['Actuals']), len(train_results_war_df['Predictions']))
    dates = dates[:min_length]

    plt.figure(figsize=(12, 6))
    plt.semilogy(dates, train_results_war_df['Actuals'][:min_length], label='Actual GDP', color='orange')
    plt.semilogy(dates, train_results_war_df['Predictions'][:min_length], label='GDP Predictions Of Model With War Variable On Training', color='red')

    patches = []  
    for war, (start, end) in war_periods.items():
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        if end > start_date:
            plt.axvspan(max(start, start_date), min(end, end_date), color=war_colors[war], alpha=0.5)
            patches.append(mpatches.Patch(color=war_colors[war], label=war))


    plt.title('Prediction Of GDP On Train Data (Log Scale)')
    plt.xlabel('Date')
    plt.ylabel('Predicted Value (Log Scale)')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(10))
    plt.xlim(start_date, end_date)
    war_patches = [mpatches.Patch(color=color, label=war) for war, color in war_colors.items()]

    gdp_patch = [mpatches.Patch(color='orange', label='Actual GDP'),
             mpatches.Patch(color='red', label='GDP Predictions Of Model With War Variable On Training')]
    all_patches = war_patches + gdp_patch
    plt.gca().add_artist(plt.legend(handles=gdp_patch, loc='upper left'))
    plt.gca().add_artist(plt.legend(handles=war_patches, loc='lower right'))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_failure_analysis_covid(test_results_war_df):

    failure_periods = {
        "COVID-19": ("2020-01-01", "2022-06-30")
    }

    num_samples = len(test_results_war_df)
    dates = pd.date_range(start='2015-12-31', periods=num_samples, freq='Q')
    plt.figure(figsize=(12, 6))
    plt.plot(dates, test_results_war_df['Actuals'], label='Actual GDP', marker='x',color='orange')
    plt.plot(dates, test_results_war_df['Predictions'], label='GDP Predictions of Model', color='red', marker='o')
    start_date = pd.Timestamp('2015-12-31')
    for war, (start, end) in failure_periods.items():
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        if end > start_date:
            plt.axvspan(max(start, start_date), end, color='lightblue', alpha=0.3)

    plt.title('Prediction failure during COVID19')
    plt.xlabel('Date')
    plt.ylabel('Predicted Value')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45, ha='right')

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

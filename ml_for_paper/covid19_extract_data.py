from tensorflow.keras.optimizers import Nadam
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import tensorflow.keras.layers as KL
from datetime import timedelta
import numpy as np
import pandas as pd
import tensorflow.keras.backend as K

import os

from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.linear_model import LinearRegression, Ridge

import datetime
import gc
from tqdm import tqdm

import xgboost as xgb

import lightgbm as lgb

from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

pd.options.display.max_rows = 500
pd.options.display.max_columns = 500

PATH_TRAIN = "../ml_inputs/TrainMaster.csv"
df_data = pd.read_csv(PATH_TRAIN)

#===============================================================================================#
# 1. Confirmed case pattern analysis
#===============================================================================================#
g = df_data.groupby("Province_State")
for s, df in g:
    df = df.reset_index()

    cc = df['ConfirmedCases']
    cc.plot.line(figsize=(10, 5))
    plt.legend(loc='center left')
    plt.xlabel(s)

    lock_down = df[df['Date'] == df['LockDownDate1']]
    if len(lock_down) != 0:
        plt.axvline(lock_down.index[0], color='r')
        plt.text(lock_down.index[0], 10, lock_down['Date'].to_string(), fontsize=12)

    # plt.show()
    plt.savefig(f'../ml_outputs/{s}.png')
    plt.clf()

# df_over_state = pd.DataFrame(row_data)
# df_over_state.to_csv('df_over_state.csv')

## Look at all cases
# df_over_state.plot.line(figsize=(10, 5))
# plt.legend(loc='center left')
# plt.show()

## Look at one by one
# for c in df_over_state.columns.values:
#     df_over_state[c].plot.line(figsize=(10, 5))
#     plt.legend(loc='center left')
#     plt.show()

# print(df_over_state)

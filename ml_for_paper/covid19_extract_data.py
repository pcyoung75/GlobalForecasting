from tensorflow.keras.optimizers import Nadam
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import tensorflow.keras.layers as KL
from datetime import timedelta
import numpy as np
import pandas as pd
import tensorflow.keras.backend as K

import os
pd.options.display.max_rows = 500
pd.options.display.max_columns = 500


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

PATH_TRAIN = "../Data/train_final.csv"
df_data = pd.read_csv(PATH_TRAIN)

#===============================================================================================#
# 1. Confirmed case pattern analysis
#===============================================================================================#
g = df_data.groupby("Province_State")["ConfirmedCases"]
row_data = {}
for s, v in g:
    row_data[s] = v.tolist()

df_over_state = pd.DataFrame(row_data)

## Look at all cases
# df_over_state.plot.line(figsize=(10, 5))
# plt.legend(loc='center left')
# plt.show()

## Look at one by one
# for c in df_over_state.columns.values:
#     df_over_state[c].plot.line(figsize=(10, 5))
#     plt.legend(loc='center left')
#     plt.show()

print(df_over_state)

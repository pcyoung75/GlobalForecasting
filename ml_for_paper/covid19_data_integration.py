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

PATH_TRAIN = "../ml_inputs/TrainMaster_final.csv"
PATH_REGION_METADATA = "../ml_inputs/region_metadata.csv"
PATH_RESULT_DATA = "../ml_inputs/TrainMaster_final_v3.csv"

data = pd.read_csv(PATH_TRAIN)
region_metadata = pd.read_csv(PATH_REGION_METADATA)

df = data.merge(region_metadata, on=["Country_Region", "Province_State"])

print(df)

df.to_csv(PATH_RESULT_DATA)

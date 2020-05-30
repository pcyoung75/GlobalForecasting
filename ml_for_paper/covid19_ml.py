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

import warnings
warnings.filterwarnings("ignore")


class Covid19ML():
    def __init__(self):
        self.days_since_cases = None
        self.seed = 42
        self.n_dates_train = None
        self.n_dates_test = None
        self.val_days = None
        self.mad_factor = None
        self.lags = 41

    def rmse(self, yt, yp):
        # RMSE: Root Mean Square Error
        # return np.sqrt(np.mean((yt - yp) ** 2))
        return np.sqrt(mean_squared_error(np.log1p(yt), np.log1p(yp)))

    ## function for building and predicting using LGBM model
    def build_predict_lgbm(self, df_train, df_test, gap):
        LGB_PARAMS = {"objective": "regression",
                      "num_leaves": 5,
                      "learning_rate": 0.013,
                      "bagging_fraction": 0.91,
                      "feature_fraction": 0.81,
                      "reg_alpha": 0.13,
                      "reg_lambda": 0.13,
                      "metric": "rmse",
                      "seed": self.seed,
                      'verbose': -1
                      }

        df_train.dropna(subset=["target_cc", "target_ft", f"lag_{gap}_cc", f"lag_{gap}_ft"], inplace=True)

        target_cc = df_train.target_cc
        target_ft = df_train.target_ft

        test_lag_cc = df_test[f"lag_{gap}_cc"].values
        test_lag_ft = df_test[f"lag_{gap}_ft"].values

        df_train.drop(["target_cc", "target_ft"], axis=1, inplace=True)
        df_test.drop(["target_cc", "target_ft"], axis=1, inplace=True)

        categorical_features = ["continent"]

        dtrain_cc = lgb.Dataset(df_train, label=target_cc, categorical_feature=categorical_features)
        dtrain_ft = lgb.Dataset(df_train, label=target_ft, categorical_feature=categorical_features)

        model_cc = lgb.train(LGB_PARAMS, train_set=dtrain_cc, num_boost_round=200)
        model_ft = lgb.train(LGB_PARAMS, train_set=dtrain_ft, num_boost_round=200)

        # inverse transform from log of change from last known value
        y_pred_cc = np.expm1(model_cc.predict(df_test, num_boost_round=200)) + test_lag_cc
        y_pred_ft = np.expm1(model_ft.predict(df_test, num_boost_round=200)) + test_lag_ft

        return y_pred_cc, y_pred_ft, model_cc, model_ft

    ## function for predicting moving average decay model
    def predict_mad(self, df_test, gap, val=False):
        # Validation days to evaluate ML models

        df_test["avg_diff_cc"] = (df_test[f"lag_{gap}_cc"] - df_test[f"lag_{gap + 3}_cc"]) / 3
        df_test["avg_diff_ft"] = (df_test[f"lag_{gap}_ft"] - df_test[f"lag_{gap + 3}_ft"]) / 3

        if val:
            y_pred_cc = df_test[f"lag_{gap}_cc"] + gap * df_test.avg_diff_cc - (
                        1 - self.mad_factor) * df_test.avg_diff_cc * np.sum([x for x in range(gap)]) / self.val_days
            y_pred_ft = df_test[f"lag_{gap}_ft"] + gap * df_test.avg_diff_ft - (
                        1 - self.mad_factor) * df_test.avg_diff_ft * np.sum([x for x in range(gap)]) / self.val_days
        else:
            y_pred_cc = df_test[f"lag_{gap}_cc"] + gap * df_test.avg_diff_cc - (
                        1 - self.mad_factor) * df_test.avg_diff_cc * np.sum([x for x in range(gap)]) / self.n_dates_test
            y_pred_ft = df_test[f"lag_{gap}_ft"] + gap * df_test.avg_diff_ft - (
                        1 - self.mad_factor) * df_test.avg_diff_ft * np.sum([x for x in range(gap)]) / self.n_dates_test

        return y_pred_cc, y_pred_ft
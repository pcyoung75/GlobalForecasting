import lightgbm as lgb
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error


class LightGBM():
    def __init__(self):
        SEED = 42
        self.num_round = 15000
        self.col_cat = []
        self.params = {'num_leaves': 8,
                      'min_data_in_leaf': 5,  # 42,
                      'objective': 'regression',
                      'max_depth': 8,
                      'learning_rate': 0.02,
                      'boosting': 'gbdt',
                      'bagging_freq': 5,  # 5
                      'bagging_fraction': 0.8,  # 0.5,
                      'feature_fraction': 0.8201,
                      'bagging_seed': SEED,
                      'reg_alpha': 1,  # 1.728910519108444,
                      'reg_lambda': 4.9847051755586085,
                      'random_state': SEED,
                      'metric': 'mse',
                      'verbosity': 100,
                      'min_gain_to_split': 0.02,  # 0.01077313523861969,
                      'min_child_weight': 5,  # 19.428902804238373,
                      'num_threads': 6,
                      'verbose': -1
                      }

    def fit(self, X_train, y_train):
        # X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=self.col_cat)
        valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=self.col_cat)

        self.model = lgb.train(self.params, train_data, self.num_round, valid_sets=[train_data, valid_data],
                                verbose_eval=False,
                                early_stopping_rounds=150)

        self.best_itr = self.model.best_iteration

    def predict(self, X_test):
        return self.model.predict(X_test)

    def feature_importance(self):
        # display feature importance
        col_var2 = []
        tmp = pd.DataFrame()
        tmp["feature"] = col_var2
        tmp["importance"] = self.model.feature_importance()
        tmp = tmp.sort_values('importance', ascending=False)
        print(tmp)
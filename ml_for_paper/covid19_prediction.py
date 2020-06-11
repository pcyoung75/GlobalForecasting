import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import warnings
import winsound
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.linear_model import LinearRegression, Ridge

import datetime
import gc
from tqdm import tqdm

import xgboost as xgb

import lightgbm as lgb

warnings.filterwarnings("ignore")


class Covid19Predictor():
    def __init__(self):
        super().__init__()
        self.days_since_cases = [1, 10, 50, 100, 500, 1000, 5000, 10000]
        self.val_days = 7
        self.mad_factor = 0.5
        self.lags = 13
        self.seed = 42
        self.n_dates_train = None
        self.n_dates_test = None

    def load_data(self, path_train, path_test):
        # Load data
        self.train = pd.read_csv(path_train)
        self.test = pd.read_csv(path_test)

    def load_extra_data(self, path_region_metadata, path_region_date_metadata):
        self.region_metadata = pd.read_csv(path_region_metadata)
        self.region_date_metadata = pd.read_csv(path_region_date_metadata)

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

        df_train.dropna(subset=["target_cc", f"lag_{gap}_cc"], inplace=True)

        target_cc = df_train.target_cc

        test_lag_cc = df_test[f"lag_{gap}_cc"].values

        df_train.drop(["target_cc"], axis=1, inplace=True)
        df_test.drop(["target_cc"], axis=1, inplace=True)

        # df_train.to_csv('temp.csv')
        # dtrain_cc = lgb.Dataset(df_train, label=target_cc, categorical_feature=categorical_features)
        dtrain_cc = lgb.Dataset(df_train, label=target_cc)

        # model_cc = lgb.train(LGB_PARAMS, train_set=dtrain_cc, num_boost_round=200)
        model_cc = lgb.train(LGB_PARAMS, train_set=dtrain_cc)

        # inverse transform from log of change from last known value
        pre_results = model_cc.predict(df_test, num_boost_round=200)
        # pre_results = model_cc.predict(df_test)

        y_pred_cc = np.expm1(pre_results) + test_lag_cc
        # y_pred_cc = pre_results

        return y_pred_cc, model_cc

    ## function for predicting moving average decay model
    def predict_mad(self, df_test, gap, val=False):
        # Validation days to evaluate ML models

        df_test["avg_diff_cc"] = (df_test[f"lag_{gap}_cc"] - df_test[f"lag_{gap + 3}_cc"]) / 3

        if val:
            y_pred_cc = df_test[f"lag_{gap}_cc"] + gap * df_test.avg_diff_cc - (
                        1 - self.mad_factor) * df_test.avg_diff_cc * np.sum([x for x in range(gap)]) / self.val_days
        else:
            y_pred_cc = df_test[f"lag_{gap}_cc"] + gap * df_test.avg_diff_cc - (
                        1 - self.mad_factor) * df_test.avg_diff_cc * np.sum([x for x in range(gap)]) / self.n_dates_test

        return y_pred_cc

    def prepare_data(self):
        # Combine training data and test data
        self.train = self.train.merge(self.test[["ForecastId", "Province_State", "Country_Region", "Date"]],
                            on=["Province_State", "Country_Region", "Date"], how="left")

        # Take the test data that do not belong to the training data
        self.test = self.test[~self.test.Date.isin(self.train.Date.unique())]

        # Make a whole data
        self.df_panel = pd.concat([self.train, self.test], sort=False)

        # Combine state and country into 'geography'
        self.df_panel["geography"] = self.df_panel.Country_Region.astype(str) + ": " + self.df_panel.Province_State.astype(str)

        # merging external metadata
        self.df_panel = self.df_panel.merge(self.region_metadata, on=["Country_Region", "Province_State"])
        self.df_panel = self.df_panel.merge(self.region_date_metadata, on=["Country_Region", "Province_State", "Date"], how="left")

        # Change a common date time
        self.df_panel.Date = pd.to_datetime(self.df_panel.Date, format="%Y-%m-%d")

        # Sort
        self.df_panel.sort_values(["geography", "Date"], inplace=True)

        self.min_date_train = np.min(self.df_panel[~self.df_panel.Id.isna()].Date)
        self.max_date_train = np.max(self.df_panel[~self.df_panel.Id.isna()].Date)

        self.min_date_test = np.min(self.df_panel[~self.df_panel.ForecastId.isna()].Date)
        self.max_date_test = np.max(self.df_panel[~self.df_panel.ForecastId.isna()].Date)

        self.n_dates_train = len(self.df_panel[~self.df_panel.Id.isna()].Date.unique())
        self.n_dates_train = len(self.df_panel[~self.df_panel.ConfirmedCases.isna()].Date.unique())
        self.n_dates_test = len(self.df_panel[~self.df_panel.ForecastId.isna()].Date.unique())

        print(f"Number of dates for training: {self.n_dates_train}")
        print(f"Number of dates for test: {self.n_dates_test}")

        print("Train date range:", str(self.min_date_train), " - ", str(self.max_date_train))
        print("Test date range:", str(self.min_date_test), " - ", str(self.max_date_test))

    def perform_feature_engineering(self):
        # creating lag features
        for lag in range(1, self.lags):
            self.df_panel[f"lag_{lag}_cc"] = self.df_panel.groupby("geography")["ConfirmedCases"].shift(lag)
            # self.df_panel[f"lag_{lag}_rc"] = self.df_panel.groupby("geography")["Recoveries"].shift(lag)

        for case in self.days_since_cases:
            self.df_panel = self.df_panel.merge(self.df_panel[self.df_panel.ConfirmedCases >= case].groupby("geography")["Date"].min().reset_index().rename(columns = {"Date": f"case_{case}_date"}), on = "geography", how = "left")

    def prepare_features(self, df, gap):
        # Percentage for the confirmed case over population
        df["perc_1_cc"] = df[f"lag_{gap}_cc"] / df.population

        # Difference between the confirmed case and the previous confirmed case (i.e., new cases)
        df["diff_1_cc"] = df[f"lag_{gap}_cc"] - df[f"lag_{gap + 1}_cc"]
        df["diff_2_cc"] = df[f"lag_{gap + 1}_cc"] - df[f"lag_{gap + 2}_cc"]
        df["diff_3_cc"] = df[f"lag_{gap + 2}_cc"] - df[f"lag_{gap + 3}_cc"]

        df["diff_123_cc"] = (df[f"lag_{gap}_cc"] - df[f"lag_{gap + 3}_cc"]) / 3

        # Change rate between the new cases and the previous new cases
        df["diff_change_1_cc"] = df.diff_1_cc / df.diff_2_cc
        df["diff_change_2_cc"] = df.diff_2_cc / df.diff_3_cc

        df["diff_change_12_cc"] = (df.diff_change_1_cc + df.diff_change_2_cc) / 2

        # Change rate between the confirmed cases and the previous confirmed cases
        df["change_1_cc"] = df[f"lag_{gap}_cc"] / df[f"lag_{gap + 1}_cc"]
        df["change_2_cc"] = df[f"lag_{gap + 1}_cc"] / df[f"lag_{gap + 2}_cc"]
        df["change_3_cc"] = df[f"lag_{gap + 2}_cc"] / df[f"lag_{gap + 3}_cc"]

        df["change_1_3_cc"] = df[f"lag_{gap}_cc"] / df[f"lag_{gap + 3}_cc"]

        df["change_1_7_cc"] = df[f"lag_{gap}_cc"] / df[f"lag_{gap + 7}_cc"]

        # days since the cases of 1, 10, 50, and ... occurs
        for case in self.days_since_cases:
            df[f"days_since_{case}_case"] = (df[f"case_{case}_date"] - df.Date).astype("timedelta64[D]")
            df.loc[df[f"days_since_{case}_case"] < gap, f"days_since_{case}_case"] = np.nan

        # target variable is log of change from last known value
        df["target_cc"] = np.log1p(df.ConfirmedCases - df[f"lag_{gap}_cc"])
        df.to_csv('temp.csv')
        features = [
            f"lag_{gap}_cc",
            # f"lag_{gap}_rc",
            # "perc_1_cc",
            # "diff_1_cc",
            # "diff_2_cc",
            # "diff_3_cc",
            # "diff_123_cc",
            # "diff_change_1_cc",
            # "diff_change_2_cc",
            # "diff_change_12_cc",
            # "change_1_cc",
            # "change_2_cc",
            # "change_3_cc",
            # "change_1_3_cc",
            # "change_1_7_cc",
            # "days_since_1_case",
            # "days_since_10_case",
            # "days_since_50_case",
            # "days_since_100_case",
            # "days_since_500_case",
            # "days_since_1000_case",
            # "days_since_5000_case",
            # "days_since_10000_case",
            # "lat",
            # "lon",
            # "population",
            # "area",
            # "density",
            "target_cc"
        ]

        return df[features]

    def perform_ML(self):
        df_train = self.df_panel[~self.df_panel.Id.isna()]
        df_test_full = self.df_panel[~self.df_panel.ForecastId.isna()]

        df_preds_val = []
        df_preds_test = []

        for date in df_test_full.Date.unique():

            print("Processing date:", date)

            # ignore date already present in train data
            if date in df_train.Date.values:
                df_pred_test = df_test_full.loc[
                    df_test_full.Date == date, ["ForecastId", "ConfirmedCases"]].rename(
                    columns={"ConfirmedCases": "ConfirmedCases_test"})

                # multiplying predictions by 41 (=self.lags) to not look cool on public LB
                df_pred_test.ConfirmedCases_test = df_pred_test.ConfirmedCases_test * self.lags
            else:
                df_test = df_test_full[df_test_full.Date == date]

                gap = (pd.Timestamp(date) - self.max_date_train).days

                if gap <= self.val_days:
                    val_date = self.max_date_train - pd.Timedelta(self.val_days, "D") + pd.Timedelta(gap, "D")

                    df_build = df_train[df_train.Date < val_date]
                    df_val = df_train[df_train.Date == val_date]

                    X_build = self.prepare_features(df_build, gap)
                    X_val = self.prepare_features(df_val, gap)

                    y_val_cc_lgb, _ = self.build_predict_lgbm(X_build, X_val, gap)
                    y_val_cc_mad = self.predict_mad(df_val, gap, val=True)

                    df_pred_val = pd.DataFrame({"Id": df_val.Id.values,
                                                "ConfirmedCases_val_lgb": y_val_cc_lgb,
                                                "ConfirmedCases_val_mad": y_val_cc_mad
                                                })

                    df_preds_val.append(df_pred_val)

                X_train = self.prepare_features(df_train, gap)
                X_test = self.prepare_features(df_test, gap)

                y_test_cc_lgb, model_cc = self.build_predict_lgbm(X_train, X_test, gap)
                y_test_cc_mad = self.predict_mad(df_test, gap)

                if gap == 1:
                    model_1_cc = model_cc
                    features_1 = X_train.columns.values
                elif gap == 14:
                    model_14_cc = model_cc
                    features_14 = X_train.columns.values
                elif gap == 28:
                    model_28_cc = model_cc
                    features_28 = X_train.columns.values

                df_pred_test = pd.DataFrame({"ForecastId": df_test.ForecastId.values,
                                             "ConfirmedCases_test_lgb": y_test_cc_lgb,
                                             "ConfirmedCases_test_mad": y_test_cc_mad
                                             })

            df_preds_test.append(df_pred_test)

        # combine predicted data for validation and predicted data for test
        self.df_panel = self.df_panel.merge(pd.concat(df_preds_val, sort=False), on="Id", how="left")
        self.df_panel = self.df_panel.merge(pd.concat(df_preds_test, sort=False), on="ForecastId", how="left")

        # calculate RMSE (Root Mean Square Error)
        rmsle_cc_lgb = self.rmse(self.df_panel[~self.df_panel.ConfirmedCases_val_lgb.isna()].ConfirmedCases,
                                 self.df_panel[~self.df_panel.ConfirmedCases_val_lgb.isna()].ConfirmedCases_val_lgb)

        rmsle_cc_mad = self.rmse(self.df_panel[~self.df_panel.ConfirmedCases_val_mad.isna()].ConfirmedCases,
                                 self.df_panel[~self.df_panel.ConfirmedCases_val_mad.isna()].ConfirmedCases_val_mad)


        print("\n#============================ Validation Results ============================#")
        print("LGB CC RMSLE Val of", self.val_days, "days for CC:", round(rmsle_cc_lgb, 2))
        print("\n")
        print("MAD CC RMSLE Val of", self.val_days, "days for CC:", round(rmsle_cc_mad, 2))

        print("\n#============================ Test Results ============================#")
        df_test = self.df_panel.loc[~self.df_panel.ForecastId.isna(), ["ForecastId", "Country_Region", "Province_State", "Date",
                                                                         "ConfirmedCases_test", "ConfirmedCases_test_lgb",
                                                                         "ConfirmedCases_test_mad"]].reset_index()

        df_test["ConfirmedCases"] = 0.13 * df_test.ConfirmedCases_test_lgb + 0.87 * df_test.ConfirmedCases_test_mad
        print(df_test)


PATH_TRAIN = "../ml_inputs/train.csv"
PATH_TEST = "../ml_inputs/test.csv"

PATH_REGION_METADATA = "../ml_inputs/region_metadata.csv"
PATH_REGION_DATE_METADATA = "../ml_inputs/region_date_metadata.csv"

c19 = Covid19Predictor()
c19.load_data(PATH_TRAIN, PATH_TEST)
c19.load_extra_data(PATH_REGION_METADATA, PATH_REGION_DATE_METADATA)
c19.prepare_data()
c19.perform_feature_engineering()
c19.perform_ML()

winsound.Beep(1000, 440)
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import warnings
from ml_for_paper.covid19_ml import Covid19ML
import winsound
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

"""
ratio:
2559	261	427.0865539
2559	261	432.6400268
2559	261	430.6773733
2559	261	426.9234473
		
difference:		
2503	262	284.3276085
2503	262	286.7755116
2503	262	284.3276085
2742	289	272.6086639
2742	289	272.1715762
 
 
시간별로 training 
인구수 로 나뉘서 scaling
"""

class Covid19Predictor(Covid19ML):
    def __init__(self):
        super().__init__()
        self.days_since_cases = [1, 10, 50, 100, 500, 1000, 5000, 10000]
        self.val_days = 14
        self.mad_factor = 0.5
        self.lags = 20
        self.mode = 'ratio'
        # self.mode = 'diff'

    def load_data(self, path_train, path_test):
        # Load data
        self.train = pd.read_csv(path_train)
        self.test = pd.read_csv(path_test)

    def load_extra_data(self, path_region_metadata):
        self.region_metadata = pd.read_csv(path_region_metadata)

    def prepare_data(self):
        # Combine training data and test data
        # self.train = self.train.merge(self.test[["ForecastId", "Province_State", "Country_Region", "Date"]],
        #                     on=["Province_State", "Country_Region", "Date"], how="left")

        # Take the test data that do not belong to the training data
        # self.test = self.test[~self.test.Date.isin(self.train.Date.unique())]

        # Make a whole data
        # self.df_panel = pd.concat([self.train, self.test], sort=False)

        # Combine state and country into 'geography'
        # self.train["Province_State"]
        # self.df_panel["geography"] = self.df_panel.Country_Region.astype(str) + ": " + self.df_panel.Province_State.astype(str)

        # merging external metadata
        self.df_panel = self.train.merge(self.region_metadata, on=["Country_Region", "Province_State"])

        # drop unnecessary columns
        self.df_panel = self.df_panel.drop(columns=['Country_Region', 'continent'])

        # Change a common date time
        self.df_panel.Date = pd.to_datetime(self.df_panel.Date, format="%Y-%m-%d")

        # Sort
        self.df_panel.sort_values(["Province_State", "Date"], inplace=True)

        # self.min_date_train = np.min(self.df_panel[~self.df_panel.Id.isna()].Date)
        # self.max_date_train = np.max(self.df_panel[~self.df_panel.Id.isna()].Date)
        #
        # self.min_date_test = np.min(self.df_panel[~self.df_panel.ForecastId.isna()].Date)
        # self.max_date_test = np.max(self.df_panel[~self.df_panel.ForecastId.isna()].Date)
        #
        # self.n_dates_train = len(self.df_panel[~self.df_panel.Id.isna()].Date.unique())
        # self.n_dates_train = len(self.df_panel[~self.df_panel.ConfirmedCases.isna()].Date.unique())
        # self.n_dates_test = len(self.df_panel[~self.df_panel.ForecastId.isna()].Date.unique())

        df_states = self.df_panel.groupby("Province_State")

        for s, df in df_states:
            print(s, ':',  np.min(df.Date), '~', np.max(df.Date))


    def perform_feature_engineering(self):
        # creating lag features
        for lag in range(1, self.lags):
            self.df_panel[f"lag_{lag}_cc"] = self.df_panel.groupby("Province_State")["ConfirmedCases"].shift(lag)

        for lag in range(1, self.lags):
            if lag == 1:
                self.df_panel[f"lag_{lag}_ratio_cc"] = self.df_panel[f"ConfirmedCases"] / self.df_panel[f"lag_1_cc"]
            else:
                self.df_panel[f"lag_{lag}_ratio_cc"] = self.df_panel[f"lag_{lag-1}_cc"]/self.df_panel[f"lag_{lag}_cc"]

        for case in self.days_since_cases:
            self.df_panel = self.df_panel.merge(self.df_panel[self.df_panel.ConfirmedCases >= case].groupby("Province_State")["Date"].min().reset_index().rename(columns = {"Date": f"case_{case}_date"}), on = "Province_State", how = "left")

    def prepare_features_ratio(self, df, lags):
        # Percentage for the confirmed case over population
        for l in range(1, lags-2):
            # Percentage for the confirmed case over population
            df[f"perc_{l}_cc"] = df[f"lag_{l}_cc"] / df.population
            df[f"perc_{l}_ratio_cc"] = df[f"lag_{l}_ratio_cc"] / df.population

            # Difference between the confirmed case and the previous confirmed case (i.e., new cases)
            df[f"diff_{l}_cc"] = df[f"lag_{l}_cc"] - df[f"lag_{l + 1}_cc"]

            # Change rate between the confirmed cases and the previous confirmed cases
            df[f"change_{l}_cc"] = df[f"lag_{l}_cc"] / df[f"lag_{l + 1}_cc"]

            # df["diff_123_cc"] = (df[f"lag_{gap}_cc"] - df[f"lag_{gap + 3}_cc"]) / 3
            df[f"diff_1_{l+1}_cc"] = (df[f"lag_{1}_cc"] - df[f"lag_{l + 2}_cc"]) / (l+1)

            # df["change_1_7_cc"] = df[f"lag_{gap}_cc"] / df[f"lag_{gap + 7}_cc"]
            df[f"change_1_{l+1}_cc"] = df[f"lag_{1}_cc"] / df[f"lag_{l + 2}_cc"]

        # days since the cases of 1, 10, 50, and ... occurs
        for case in self.days_since_cases:
            df[f"days_since_{case}_case"] = (df[f"case_{case}_date"] - df.Date).astype("timedelta64[D]")
            df.loc[df[f"days_since_{case}_case"] < 1, f"days_since_{case}_case"] = np.nan

        # My
        # target variable is log of change from last known value
        # df["target_cc"] = np.log1p(df.ConfirmedCases - df[f"lag_{gap}_cc"])
        if self.mode == 'diff':
            df["target_cc"] = df.ConfirmedCases - df[f"lag_{1}_cc"] # use the difference
        elif self.mode == 'ratio':
            df["target_cc"] = df[f"lag_{1}_ratio_cc"] # use the ratio
        # df["target_cc"] = df.ConfirmedCases

        # df.to_csv('temp.csv')

        features = [
            # "diff_1_3_cc",
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

        for l in range(1, lags-2):
            if self.mode == 'diff':
                features.append(f"lag_{l}_cc")
            elif self.mode == 'ratio':
                features.append(f"lag_{l}_ratio_cc")
                features.append(f"lag_{l}_cc")
                # features.append(f"perc_{l}_ratio_cc")
            # features.append(f"perc_{l}_cc") # Strong
            # features.append(f"diff_{l}_cc")
            # features.append(f"change_{l}_cc")

        if self.mode == 'diff':
            pass
        elif self.mode == 'ratio':
            features.remove(f"lag_1_ratio_cc")
        # features.remove(f"lag_1_cc")

        return df[features]

    def perform_ML(self):
        lags = 8
        train_days = 8
        test_days = 1
        # # Take data according to states
        df_states = self.df_panel.groupby("Province_State")

        RMSEs = 0

        for s, df in df_states:
            df_train = df[:train_days]
            df_test = df[train_days:train_days + test_days]

            df_train = self.prepare_features_ratio(df_train, lags)
            df_test = self.prepare_features_ratio(df_test, lags)

            # Remove rows which have the zero CC
            # df_test = df_test[df_test.target_cc != 0]
            # df_train = df_train[df_train.target_cc != 0]

            # Remove rows which have the nan value
            # x_val = x_val[~x_val.target_cc.isna()]
            # x_train = x_train[~x_train.target_cc.isna()]
            #
            # x_val = x_val[x_val.target_cc != float('inf')]
            # x_train = x_train[x_train.target_cc != float('inf')]

            # if self.mode == 'diff':
            #     pass
            # elif self.mode == 'ratio':
            #     for i in range(2, 8):
            #         x_val = x_val[x_val[f"lag_{i}_ratio_cc"] != float('inf')]
            #         x_train = x_train[x_train[f"lag_{i}_ratio_cc"] != float('inf')]

            if s == 'New York':
                print('')

            actual_cc = df_test.target_cc
            # perform ML
            y_test_cc_lgb, model_cc = self.build_predict_lgbm(df_train, df_test)
            # y_test_cc_mad = self.predict_mad(df_test, gap)

            # get CC from the ratio
            if self.mode == 'diff':
                actual_cc = actual_cc + df_test.lag_1_cc
                y_test_cc_lgb = y_test_cc_lgb + df_test.lag_1_cc
            elif self.mode == 'ratio':
                actual_cc = actual_cc * df_test.lag_1_cc
                y_test_cc_lgb = y_test_cc_lgb * df_test.lag_1_cc

            df_temp = pd.DataFrame({"Actual_CC": actual_cc,
                                    "LGB_CC": y_test_cc_lgb
                                    })

            # calculate RMSE (Root Mean Square Error)
            df_temp = df_temp[df_temp.Actual_CC != float('inf')]
            rmsle_cc_lgb = self.rmse(df_temp[~df_temp.Actual_CC.isna()].Actual_CC,
                                     df_temp[~df_temp.Actual_CC.isna()].LGB_CC)
            RMSEs += rmsle_cc_lgb
            print(f'{s} \t {len(df_train)} \t {len(df_test)} \t {rmsle_cc_lgb}')

        print('Average RMSE:', RMSEs/len(df_states))


PATH_TRAIN = "../ml_inputs/TrainMaster.csv"
PATH_TEST = "../ml_inputs/test.csv"
PATH_REGION_METADATA = "../ml_inputs/region_metadata.csv"

c19 = Covid19Predictor()
c19.load_data(PATH_TRAIN, PATH_TEST)
c19.load_extra_data(PATH_REGION_METADATA)
c19.prepare_data()
c19.perform_feature_engineering()
c19.perform_ML()

winsound.Beep(1000, 440)
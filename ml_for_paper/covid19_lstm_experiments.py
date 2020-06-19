import os
from time_series_ml_v2 import TimeSeriesML
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
import warnings
import winsound
from saveResults import SaveResults
from datetime import datetime
from statistics import stdev
from statistics import mean

warnings.filterwarnings('ignore')


"""
"Alabama", "Alaska", "Arizona", "Arkansas",  "California", "Colorado", "Connecticut", "Delaware", "District of Columbia",
"Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine",
"Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire",
"New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania",
"Puerto Rico", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
"West Virginia", "Wisconsin", "Wyoming"
"""


class Covid19Predictor():
    def __init__(self):
        # fix random seed for reproducibility
        np.random.seed(7)
        self.cf = {}
        # self.cf['filename'] = os.path.join('../ml_inputs', 'covid19_montana.csv')
        # self.cf['filename'] = os.path.join('../ml_inputs', 'covid19.csv')
        self.cf['filename'] = os.path.join('../ml_inputs', 'TrainMaster_final.csv')
        # self.cf['filename'] = os.path.join('../ml_inputs', 'covid19_three.csv')
        # self.cf['learnedfile'] = os.path.join('saved_models', 'covid19_three.hdf5')
        self.cf['date_data'] = ['Date']
        self.cf['sort'] = None
        # self.cf['look_back'] = 1
        self.cf['look_back'] = 20

        # ===============================================
        self.lags = 4
        # self.days_since_cases = [1, 10, 50, 100, 500, 1000, 5000, 10000]
        self.days_since_cases = [1, 10, 50]
        self.val_days = 14
        self.mode = 'ratio'

        # Excel
        self.excel = {}
        self.excel['X Variables'] = []
        self.excel['Y Target'] = []
        self.excel['Train States'] = []
        self.excel['Target State'] = []
        self.excel['avg_score'] = []
        self.excel['std_score'] = []
        self.excel['Train Size'] = []
        self.excel['Test Size'] = []
        self.excel['Repetition'] = []

    def save_excel_file(self):
        experiment_time = datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
        excel_file = f'../ml_outputs/{experiment_time}.xlsx'
        excel_experiment = SaveResults(excel_file)
        for k, l in self.excel.items():
            excel_experiment.insert(k, l)
        excel_experiment.save()

    def set_experiment_settings(self, train_states, test_state, x_variables):
        # self.cf['columns'] = ['lag_1_ratio_cc']
        # self.cf['columns'] = ['lag_1_ratio_cc', 'SeniorPopulation', 'FoodStamp', 'NoHealthIns', 'PovertyLevel', 'MeanTravelTime']
        # self.cf['columns'] = ['lag_1_ratio_cc', 'SeniorMalePopulation', 'PublicTransportationP', 'Household', 'Income']
        # self.cf['columns'] = ['lag_1_ratio_cc', 'Income']
        # self.cf['columns'] = ['lag_1_ratio_cc', 'SeniorPopulation', 'FoodStamp', 'NoHealthIns', 'PovertyLevel',
        #                       'MeanTravelTime', 'SeniorMalePopulation', 'PublicTransportationP', 'Household', 'Income']
        self.cf['columns'] = x_variables

        self.cf['target'] = 'lag_1_ratio_cc'
        # self.cf['columns'] = ['diff_1_cc']
        # self.cf['target'] = 'diff_1_cc'

        self.cf['train_test_split'] = None
        self.cf['train_columns'] = {'Province_State': train_states}
        # self.cf['train_columns'] = {'Province_State': ['Hawaii', 'Alaska']}
        self.cf['test_columns'] = {'Province_State': test_state}
        # self.cf['test_columns'] = {'Province_State': ['Montana']}
        # self.cf['epochs'] = 500 # better
        # self.cf['epochs'] = 100
        # self.cf['epochs'] = 20
        # self.cf['epochs'] = 40
        self.cf['epochs'] = 2
        self.cf['batch_size'] = 1
        # self.cf['batch_size'] = 32
        self.cf['save_dir'] = 'saved_models'
        self.cf['loss'] = 'mean_squared_error'
        self.cf['optimizer'] = 'adam'
        # self.cf['optimizer'] = 'rmsprop'
        # self.cf['optimizer'] = 'sgd'
        # self.cf['optimizer'] = 'adagrad'
        # self.cf['optimizer'] = 'adadelta'
        # self.cf['optimizer'] = 'adamax'
        # self.cf['optimizer'] = 'nadam'
        # self.cf['optimizer'] = 'tfoptimizer'
        self.cf['neurons'] = 50
        # self.cf['neurons'] = 200
        # self.cf['neurons'] = 4
        self.cf['layers'] = []
        # self.cf['metrics'] = 'RMSE'
        self.cf['metrics'] = 'MAE'

    def prepare_data(self, model):
        df = model.excel_data

        # As the number of lookback, make previous virtual data
        # e.g.,)
        # Date	        ConfirmedCases  ->  Date	        ConfirmedCases
        # 3/13/2020	    5                   3/11/2020	    0.001           > lookback 2
        #                                   3/12/2020	    0.001           > lookback 1
        #                                   3/13/2020	    5
        states = df.groupby("Province_State")
        states_df = []
        for s, s_df in states:
            s_df = s_df.reset_index()
            for b in range(1, self.cf['look_back'] + 1):
                first_row = s_df.iloc[0:1].copy()
                first_row['Date'][0] = first_row['Date'][0].to_datetime64() - np.timedelta64(1,'D')
                first_row['ConfirmedCases'] = 1
                s_df = pd.concat([first_row, s_df])

            states_df.append(s_df)

        df = pd.concat(states_df)
        df = df.reset_index()

        # creating lag features
        for lag in range(1, self.lags):
            # df[f"lag_{lag}_cc"] = df.groupby("Province_State")["ConfirmedCases"].shift(lag)
            df[f"lag_{lag}_cc"] = df["ConfirmedCases"].shift(lag)

        for lag in range(1, self.lags):
            if lag == 1:
                df[f"lag_{lag}_ratio_cc"] = df[f"ConfirmedCases"] / df[f"lag_1_cc"]
            else:
                df[f"lag_{lag}_ratio_cc"] = df[f"lag_{lag - 1}_cc"] / df[f"lag_{lag}_cc"]

        for l in range(1, self.lags - 2):
            # Difference between the confirmed case and the previous confirmed case (i.e., new cases)
            df[f"diff_{l}_cc"] = df[f"lag_{l}_cc"] - df[f"lag_{l + 1}_cc"]

        # for case in self.days_since_cases:
        #     df = df.merge(df[df.ConfirmedCases >= case].groupby("Province_State")[
        #                 "Date"].min().reset_index().rename(columns={"Date": f"case_{case}_date"}), on="Province_State",
        #                 how="left")

        # Percentage for the confirmed case over population
        # for l in range(1, self.lags - 2):
        #     # Percentage for the confirmed case over population
        #     df[f"perc_{l}_cc"] = df[f"lag_{l}_cc"] / df.population
        #     df[f"perc_{l}_ratio_cc"] = df[f"lag_{l}_ratio_cc"] / df.population

            # Difference between the confirmed case and the previous confirmed case (i.e., new cases)
            # df[f"diff_{l}_cc"] = df[f"lag_{l}_cc"] - df[f"lag_{l + 1}_cc"]

            # # Change rate between the confirmed cases and the previous confirmed cases
            # df[f"change_{l}_cc"] = df[f"lag_{l}_cc"] / df[f"lag_{l + 1}_cc"]
            #
            # # df["diff_123_cc"] = (df[f"lag_{gap}_cc"] - df[f"lag_{gap + 3}_cc"]) / 3
            # df[f"diff_1_{l + 1}_cc"] = (df[f"lag_{1}_cc"] - df[f"lag_{l + 2}_cc"]) / (l + 1)
            #
            # # df["change_1_7_cc"] = df[f"lag_{gap}_cc"] / df[f"lag_{gap + 7}_cc"]
            # df[f"change_1_{l + 1}_cc"] = df[f"lag_{1}_cc"] / df[f"lag_{l + 2}_cc"]

        # # days since the cases of 1, 10, 50, and ... occurs
        # for case in self.days_since_cases:
        #     df[f"days_since_{case}_case"] = (df[f"case_{case}_date"] - df.Date).astype("timedelta64[D]")
        #     df.loc[df[f"days_since_{case}_case"] < 1, f"days_since_{case}_case"] = np.nan

        # For the ratio target, remove the first row in a state data, since they are not calculated correctly.
        df = df[df['lag_1_ratio_cc'] >= 1.0]

        model.excel_data = df

    def run(self):
        model = TimeSeriesML(self.cf)
        model.load_data()
        self.prepare_data(model)

        # ======================================================================= #
        #       perform ML several times
        # ======================================================================= #
        states = ["Alabama", "Alaska", "Arizona"]
        # states = ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware",
        #           "District of Columbia", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky",
        #           "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana",
        #           "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma",
        #           "Oregon", "Pennsylvania", "Puerto Rico", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah",
        #           "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"]

        variables = ['lag_1_ratio_cc', 'SeniorPopulation', 'FoodStamp', 'NoHealthIns', 'PovertyLevel',
                     'MeanTravelTime', 'SeniorMalePopulation', 'PublicTransportationP', 'Household', 'Income']

        num_experiments = 2

        # for ex in range(num_experiments + 1):
        for st in states:
            # Make the experiment settings
            # Select training states and a target state
            train_states = states.copy()
            train_states.remove(st)
            test_state = []
            test_state.append(st)

            # Select x variables
            x_variables = ['lag_1_ratio_cc']

            # Set experiment settings
            self.set_experiment_settings(train_states, test_state, x_variables)

            # Split data
            model.split_data()

            scores = []
            for ex in range(num_experiments):
                # Perform training
                model.train()

                # Perform prediction
                y_predict, y_test = model.predict()

                # Remove the first day of CC, since we can't predict it
                first_cc = np.where(y_test > 1.0)[0][0]
                y_predict, y_test = y_predict[first_cc + 1:], y_test[first_cc + 1:]

                print(y_test)
                print(y_predict)

                # Get score of the prediction
                scores.append(model.score(y_test, y_predict))

            # Get the average score and STD
            self.excel['avg_score'].append(mean(scores))
            self.excel['std_score'].append(stdev(scores))
            self.excel['X Variables'].append(', '.join(x_variables))
            self.excel['Y Target'].append(self.cf['target'])
            self.excel['Train States'].append(', '.join(train_states))
            self.excel['Target State'].append(st)
            self.excel['Repetition'].append(num_experiments)
            self.excel['Train Size'].append(model.train_size)
            self.excel['Test Size'].append(model.test_size)

            # Show prediction results
            # model.show(y_test, y_predict)

        winsound.Beep(1000, 440)
        self.save_excel_file()


if __name__ == '__main__':
    Covid19Predictor().run()
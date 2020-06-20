import math
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import lightgbm as lgb
import warnings
import time
import threading
from multiprocessing import Process, Queue

class TimeSeriesML():

    def __init__(self, configs):
        self.cf = configs
        self.seed = 10
        self.look_back = self.cf['look_back']
        self.first_index = {}
        self.process_results = Queue()

    def load_data(self):
        self.excel_data = pd.read_csv(self.cf['filename'])
        for d in self.cf['date_data']:
            self.excel_data[d] = pd.to_datetime(self.excel_data[d])

        if self.cf['sort'] is not None:
            self.excel_data.sort_values(self.cf['sort'], inplace=True)
        return self.excel_data

    def split_data(self):
        split = self.cf['train_test_split']

        cols = self.cf['columns']

        self.dataset = self.excel_data.get(cols)
        # self.dataset.dropna(subset=cols)

        # normalize the dataset
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        if split is not None:
            scaled_df = self.scaler.fit_transform(self.dataset)
            self.dataset = pd.DataFrame(scaled_df, index=self.dataset.index, columns=self.dataset.columns)

            self.train_size = int(len(self.dataset) * split)
            self.test_size = len(self.dataset) - self.train_size

            self.df_train = self.dataset[0:self.train_size]
            self.df_test = self.dataset[self.train_size:len(self.dataset)]
        else:
            # Data which are in certain values in a column are taken as train and test data
            train_columns = self.cf['train_columns']
            test_columns = self.cf['test_columns']

            # Find train and test column keys
            train_columns_key = list(train_columns.keys())[0]
            test_columns_key = list(test_columns.keys())[0]

            # Take data which are in the values in the column
            self.dataset = self.excel_data.where(self.excel_data[train_columns_key].isin(train_columns[train_columns_key] + test_columns[test_columns_key]))

            # Select data in the selected columns (i.e., X and Y variables)
            self.dataset = self.dataset.get(cols)
            self.dataset = self.dataset.dropna(subset=cols)

            # Scale the selected data
            scaled_df = self.scaler.fit_transform(self.dataset)

            # Use the scaled data to make the selected data, so that the same index is reserved
            self.dataset = pd.DataFrame(scaled_df, index=self.dataset.index, columns=self.dataset.columns)

            # Get the train and test data from the original data
            df_train = self.excel_data.where(self.excel_data[train_columns_key].isin(train_columns[train_columns_key])).dropna(subset=[train_columns_key])
            df_test = self.excel_data.where(self.excel_data[test_columns_key].isin(test_columns[test_columns_key])).dropna(subset=[test_columns_key])

            # Get the first index of a state and store it as a dict
            states = df_train.groupby(train_columns_key)
            cur_index = 0
            for s, s_df in states:
                self.first_index[cur_index] = s
                cur_index += len(s_df)

            states = df_test.groupby(test_columns_key)
            for s, s_df in states:
                self.first_index[cur_index] = s

            # Get the train and test data from the selected data
            self.df_train = self.dataset.ix[df_train.index].dropna(subset=cols)
            self.df_test = self.dataset.ix[df_test.index].dropna(subset=cols)

            self.train_size = len(self.df_train)
            self.test_size = len(self.df_test)

            self.dataset = pd.concat((self.df_train, self.df_test), sort=False)

        self.target_column_index = self.dataset.columns.get_loc(self.cf['target'])

        print(f"train_data_size: {self.train_size} test_data_size: {self.test_size}")

        # make data sets with time windows applied
        self.x_train, self.y_train = self.create_window_dataset(self.df_train)
        self.x_test, self.y_test = self.create_window_dataset(self.df_test)

        print(f"x_train_size: {len(self.x_train)} x_test_size: {len(self.x_test)}")

    def create_window_dataset(self, dataset):
        dataX, dataY = [], []

        for i in range(len(dataset) - self.look_back - 1):
            dataX.append(dataset[i:(i + self.look_back)].values)
            dataY.append(dataset.iloc[i + self.look_back, self.target_column_index])

        # reshape data ##################################
        # dataX to be [samples, time steps, features]
        samples = len(dataX)
        windows = dataX[0].shape[0]
        features = dataX[0].shape[1]
        dataX = np.reshape(dataX, (samples, windows, features))

        # dataX to np.array
        dataY = np.array(dataY)

        return dataX, dataY

    def train_with_thread(self, size=10):
        # 1. perform common machine learning
        thread = []
        for th in range(size):
            print(f'[Thread {th}] for ML starts ============================')
            t = Process(target=self.run_train_with_thread, args=([self.x_train, self.y_train, self.x_test, self.y_test, self.process_results]))
            # t = threading.Thread(target=self.run_train_with_thread,
            #                      args=([self.x_train, self.y_train, self.x_test, self.y_test]))
            # t.setDaemon(True)
            thread.append(t)

        for t in thread:
            t.start()

        for t in thread:
            t.join()

    def run_train_with_thread(self, x_train, y_train, x_test, y_test, process_results):
        # fit the LSTM network
        ts = time.time()
        model = Sequential()
        model.add(LSTM(self.cf['neurons'], input_shape=(x_train.shape[1], x_train.shape[2]), stateful=False))
        model.add(Dense(1))
        model.compile(loss=self.cf['loss'], optimizer=self.cf['optimizer'])
        model.fit(x_train, y_train, nb_epoch=self.cf['epochs'], batch_size=self.cf['batch_size'], verbose=2)

        # test the LSTM network
        y_predict = model.predict(x_test)
        y_predict = self.inv_transform(y_predict)
        y_test = self.inv_transform(y_test)

        # Remove the first day of CC, since we can't predict it
        first_cc = np.where(y_test > 1.0)[0][0]
        y_predict, y_test = y_predict[first_cc + 1:], y_test[first_cc + 1:]

        # calculate root mean squared error
        if self.cf['metrics'] == 'RMSE':
            test_score = math.sqrt(mean_squared_error(y_test, y_predict))
        elif self.cf['metrics'] == 'MAE':
            test_score = mean_absolute_error(y_test, y_predict)

        process_results.put(test_score)
        time.sleep(0.1)
        print(f'LSTM was learned!: {(time.time() - ts)}')

    def train(self, x_train=None, y_train=None):
        if x_train is not None:
            self.x_train, self.y_train = x_train, y_train
        else:
            x_train, y_train = self.x_train, self.y_train

        # create and fit the LSTM network
        ts = time.time()
        self.model = Sequential()
        self.model.add(LSTM(self.cf['neurons'], input_shape=(x_train.shape[1], x_train.shape[2]), stateful=False))
        self.model.add(Dense(1))
        self.model.compile(loss=self.cf['loss'], optimizer=self.cf['optimizer'])
        self.model.fit(x_train, y_train, nb_epoch=self.cf['epochs'],
                       batch_size=self.cf['batch_size'], verbose=2)

        print(f'LSTM was learned!: {(time.time() - ts)}')

        if 'learnedfile' in self.cf and self.cf['learnedfile'] is not None:
            self.model.save(self.cf['learnedfile'])

    def predict(self, x_test=None, y_test=None):
        if x_test is not None:
            self.x_test, self.y_test = x_test, y_test
        else:
            x_test, y_test = self.x_test, self.y_test

        y_predict = self.model.predict(x_test)

        y_predict = self.inv_transform(y_predict)
        y_test = self.inv_transform(y_test)

        return y_predict, y_test

    def score(self, y_test, y_predict):
        # calculate root mean squared error
        if self.cf['metrics'] == 'RMSE':
            test_score = math.sqrt(mean_squared_error(y_test, y_predict))
        elif self.cf['metrics'] == 'MAE':
            test_score = mean_absolute_error(y_test, y_predict)

        return test_score

    def show(self, y_test, y_predict):
        # calculate root mean squared error
        test_score = self.score(y_test, y_predict)

        title = f'Train Length[{len(self.x_train)}]'
        title += f'  Prediction Length[{len(y_predict)}]'
        title += f'  Input Windows[{self.look_back}]'
        title += f'  {self.cf["metrics"]} Score[%.4f]' % (test_score)

        print(title)

        # shift train predictions for plotting
        actuals = self.inv_transform(self.dataset)
        y_predict_plot = np.empty_like(actuals)
        y_predict_plot[:] = np.nan
        y_predict_plot[len(actuals) - 1 - len(y_predict): len(actuals) - 1] = y_predict

        train_line = np.empty_like(actuals)
        train_line[:] = np.nan
        train_line[self.look_back : len(self.x_train) + self.look_back] = 0

        # plot baseline and prediction
        fig = plt.figure()
        # fig.suptitle('test title')
        ax = fig.gca()
        ax.set_xticks(np.arange(0, len(actuals), 1))
        plt.grid(True)

        # Draw data names
        for k, v in self.first_index.items():
            plt.axvline(k, color='#ABB2B9')
            plt.text(k, 0, v, fontsize=12)

        plt.plot(actuals)
        plt.plot(y_predict_plot)
        plt.plot(train_line)
        plt.xlabel(title)
        plt.show()

    def inv_transform(self, data):
        """
        # - scaler   = the scaler object (it needs an inverse_transform method)
        # - data     = the data to be inverse transformed as a Series, ndarray, ...
        #              (a 1d object you can assign to a df column)
        # - ftName   = the name of the column to which the data belongs
        # - colNames = all column names of the data on which scaler was fit
        #              (necessary because scaler will only accept a df of the same shape as the one it was fit on)
        """
        if isinstance(data, pd.DataFrame):
            temp = data.to_numpy()
        else:
            temp = data

        col_name = self.cf['target']
        col_names = self.cf['columns']
        dummy = pd.DataFrame(np.zeros((len(temp), len(col_names))), columns=col_names)
        dummy[col_name] = temp
        dummy = pd.DataFrame(self.scaler.inverse_transform(dummy), columns=col_names)
        return dummy[col_name].values
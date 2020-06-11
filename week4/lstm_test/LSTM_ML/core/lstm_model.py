import math
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import warnings
import time

warnings.filterwarnings("ignore")


class LSTM_Model():
    def __init__(self, configs):
        self.cf = configs
        filename = self.cf['filename']
        split = self.cf['train_test_split']
        train_columns = self.cf['train_columns']
        test_columns = self.cf['test_columns']
        cols = self.cf['columns']
        self.look_back = self.cf['look_back']

        dataframe = pd.read_csv(filename)
        dataframe.sort_values(self.cf['sort'], inplace=True)
        self.dataset = dataframe.get(cols)

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
            train_columns_key = list(train_columns.keys())[0]
            test_columns_key = list(test_columns.keys())[0]

            self.dataset = dataframe.where(dataframe[train_columns_key].isin(train_columns[train_columns_key] + test_columns[test_columns_key])).dropna()
            self.dataset = dataframe.get(cols)

            scaled_df = self.scaler.fit_transform(self.dataset)
            self.dataset = pd.DataFrame(scaled_df, index=self.dataset.index, columns=self.dataset.columns)

            df_train = dataframe.where(dataframe[train_columns_key].isin(train_columns[train_columns_key])).dropna(subset=[train_columns_key])
            df_test = dataframe.where(dataframe[test_columns_key].isin(test_columns[test_columns_key])).dropna(subset=[test_columns_key])

            self.train_size = len(df_train)
            self.test_size = len(df_test)

            self.df_train = self.dataset.ix[df_train.index]
            self.df_test = self.dataset.ix[df_test.index]

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

    def train(self, x_train=None, y_train=None):
        if x_train is not None:
            self.x_train, self.y_train = x_train, y_train
        else:
            x_train, y_train = self.x_train, self.y_train
        
        # create and fit the LSTM network
        ts = time.time()
        self.model = Sequential()
        self.model.add(LSTM(self.cf['neurons'], input_shape=(x_train.shape[1], x_train.shape[2])))
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

    def show_data(self):
        # values = dataset.values
        # groups = [0, 1, 2, 3, 5, 6, 7]
        # i = 1
        # pyplot.figure()
        # for group in groups:
        #     pyplot.subplot(len(groups), 1, i)
        #     pyplot.plot(values[:, group])
        #     pyplot.title(dataset.columns[group], y=0.5, loc='right')
        #     i += 1
        # pyplot.show()
        pass

    def show(self, y_predict, y_test):
        # calculate root mean squared error
        test_score = math.sqrt(mean_squared_error(y_test, y_predict))
        title = f'Train Length[{len(self.x_train)}]'
        title += f'  Prediction Length[{len(y_predict)}]'
        title += f'  Input Windows[{self.look_back}]'
        title += f'  RMSE Score[%.2f]' % (test_score)

        print(title)

        # shift train predictions for plotting

        actuals = self.inv_transform(self.dataset)
        y_predict_plot = np.empty_like(actuals)
        y_predict_plot[:] = np.nan
        y_predict_plot[len(actuals) - 1 - len(y_predict): len(actuals) - 1] = y_predict

        train_line = np.empty_like(actuals)
        train_line[:] = np.nan
        train_line[self.look_back : len(self.x_train) + self.look_back] = 0

        # plot baseline and predictions
        fig = plt.figure()
        # fig.suptitle('test title')
        ax = fig.gca()
        ax.set_xticks(np.arange(0, len(actuals), 1))
        plt.grid(True)
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
        col_name = self.cf['target']
        col_names = self.cf['columns']
        dummy = pd.DataFrame(np.zeros((len(data), len(col_names))), columns=col_names)
        dummy[col_name] = data
        dummy = pd.DataFrame(self.scaler.inverse_transform(dummy), columns=col_names)
        return dummy[col_name].values
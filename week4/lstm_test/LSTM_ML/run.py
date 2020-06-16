import os
from core.lstm_model import LSTM_Model
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

"""
"Alabama", "Alaska", "Arizona", "Arkansas",  "California", "Colorado", "Connecticut", "Delaware", "District of Columbia",
"Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine",
"Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire",
"New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania",
"Puerto Rico", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
"West Virginia", "Wisconsin", "Wyoming"
"""

class LSTM_ML():
    def __init__(self):
        # fix random seed for reproducibility
        np.random.seed(7)
        self.cf = {}
        self.cf['filename'] = os.path.join('data', 'covid19_montana.csv')
        # self.cf['filename'] = os.path.join('data', 'covid19.csv')
        # self.cf['filename'] = os.path.join('data', 'covid19_all.csv')
        # self.cf['filename'] = os.path.join('data', 'covid19_three.csv')
        # self.cf['learnedfile'] = os.path.join('saved_models', 'covid19_three.hdf5')
        self.cf['sort'] = ['Province_State', 'Date']
        # self.cf['columns'] = ['ConfirmedCases', 'Fatalities']
        self.cf['columns'] = ['ConfirmedCases']
        self.cf['target'] = 'ConfirmedCases'
        self.cf['train_test_split'] = None
        # self.cf['train_columns'] = None
        # self.cf['test_columns'] = None
        self.cf['train_test_split'] = 0.8
        self.cf['look_back'] = 1
        # self.cf['epochs'] = 500 # better
        self.cf['epochs'] = 100
        # self.cf['epochs'] = 2
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
        # self.cf['neurons'] = 4
        self.cf['layers'] = []

    def run(self):
        # ======================================================================= #
        #       Basic ML
        # ======================================================================= #
        # model = LSTM_Model(self.cf)
        # model.train()
        # y_predict, y_test = model.predict()
        # model.show(y_predict, y_test)

        # ======================================================================= #
        #       Basic ML
        # ======================================================================= #
        model = LSTM_Model(self.cf)
        model.train()
        y_predict, y_test = model.predict()
        model.show(y_predict, y_test)


if __name__ == '__main__':
    LSTM_ML().run()
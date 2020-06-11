from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from HML_runner import HML_runner
from DMP_runner import DMP_runner
from sklearn.svm import SVR
import pandas as pd
import numpy as np
import time
import json

class RegressionML():
    def __init__(self, metrics='R2'):
        # Set metrics
        if metrics == 'R2':
            self.score = r2_score
        elif metrics == 'MAE':
            self.score = mean_absolute_error

    def show_results(self, y_predicted, y_actual=None, ML_Alg=None, cv=5):
        if y_actual is None or len(y_actual) == 0:
            return 0

        # Score metric: R^2
        """ The coefficient R^2 is defined as (1 - u/v), where u is the residual
            sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
            sum of squares ((y_true - y_true.mean()) ** 2).sum().
            The best possible score is 1.0 and it can be negative (because the
            model can be arbitrarily worse). A constant model that always
            predicts the expected value of y, disregarding the input features,
            would get a R^2 score of 0.0."""

        # Training Cross Validation Accuracy Score
        # if ML_Alg != None:
        #     val_scores = cross_val_score(ML_Alg, X_train, np.ravel(y_train), cv=cv)
        #     print(f'Training Cross Validation Score({val_scores.mean()}):', val_scores)

        # Testing Accuracy Score
        r2 = self.score(y_actual, y_predicted)
        # if len(y_actual) > 0 and len(y_predicted) > 0:
        #     print("Testing R^2 Accuracy Score:", r2)

        return r2

    ################################################################################################
    # Decision Tree Regressor #####################################################################
    def DecisionTreeRegressor_run(self, X_train, y_train, X_test, y_actual=None):
        ml_model = self.DecisionTreeRegressor_ML(X_train, y_train)
        return self.prediction(ml_model, X_test, y_actual)

    def DecisionTreeRegressor_ML(self, X_train, y_train):
        ML_Alg = DecisionTreeRegressor()
        ml_model = ML_Alg.fit(X_train, np.ravel(y_train))
        return ml_model

    ################################################################################################
    # Gaussian NB              #####################################################################
    def GaussianNB_run(self, X_train, y_train, X_test, y_actual=None):
        ml_model = self.GaussianNB_ML(X_train, y_train)
        return self.prediction(ml_model, X_test, y_actual)

    def GaussianNB_ML(self, X_train, y_train):
        # print("*** Gaussian NB ***")
        ML_Alg = GaussianNB()
        ml_model = ML_Alg.fit(X_train, np.ravel(y_train))
        return ml_model

    ################################################################################################
    # Random Forest            #####################################################################
    def RandomForestRegressor_run(self, X_train, y_train, X_test, y_actual=None):
        ml_model = self.RandomForestRegressor_ML(X_train, y_train)
        return self.prediction(ml_model, X_test, np.ravel(y_actual))

    def RandomForestRegressor_ML(self, X_train, y_train, RF_parameters=None):
        # print("*** Random Forest Regression ***")
        if RF_parameters is not None:
            ML_Alg = RandomForestRegressor(n_estimators=RF_parameters['n_estimators'])
        else:
            ML_Alg = RandomForestRegressor(n_estimators=100)
        ml_model = ML_Alg.fit(X_train, np.ravel(y_train))
        return ml_model

    ################################################################################################
    # Linear Regressor         #####################################################################
    def LinearRegressor_run(self, X_train, y_train, X_test, y_actual=None):
        ml_model = self.LinearRegressor_ML(X_train, y_train)
        return self.prediction(ml_model, X_test, y_actual)

    def LinearRegressor_ML(self, X_train, y_train):
        # print("*** Random LinearRegression Regression ***")
        ML_Alg = LinearRegression()
        ml_model = ML_Alg.fit(X_train, np.ravel(y_train))
        return ml_model

    ################################################################################################
    # Gradient Boosting        #####################################################################
    def GradientBoostingRegressor_run(self, X_train, y_train, X_test, y_actual=None):
        ml_model = self.GradientBoostingRegressor_ML(X_train, y_train)
        return self.prediction(ml_model, X_test, y_actual)

    def GradientBoostingRegressor_ML(self, X_train, y_train):
        # print("*** Gradient Boosting Regression ***")
        # ML_Alg = GradientBoostingRegressor(n_estimators=100,
        #                                    learning_rate=0.1,
        #                                    subsample=0.5,
        #                                    max_depth=1,
        #                                    random_state=0)
        ML_Alg = GradientBoostingRegressor()
        ml_model = ML_Alg.fit(X_train, np.ravel(y_train))
        return ml_model

    ################################################################################################
    # Gaussian Process         #####################################################################
    def GaussianProcessRegressor_run(self, X_train, y_train, X_test, y_actual=None):
        ml_model = self.GaussianProcessRegressor_ML(X_train, y_train)
        return self.prediction(ml_model, X_test, y_actual)

    def GaussianProcessRegressor_ML(self, X_train, y_train):
        # print("*** Gaussian Process Regression ***")
        kernel = DotProduct() + WhiteKernel()
        ML_Alg = GaussianProcessRegressor(kernel=kernel, random_state=0)
        ml_model = ML_Alg.fit(X_train, np.ravel(y_train))
        return ml_model

    ################################################################################################
    #  Support Vector Machines  ####################################################################
    def SVMRegressor_run(self, X_train, y_train, X_test, y_actual=None):
        ml_model = self.SVMRegressor_ML(X_train, y_train)
        return self.prediction(ml_model, X_test, y_actual)

    def SVMRegressor_ML(self, X_train, y_train):
        ML_Alg = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
        ml_model = ML_Alg.fit(X_train, np.ravel(y_train))
        return ml_model

    ################################################################################################
    #  Bayesian Ridge Regression ###################################################################
    def BayesianRidgeRegressor_run(self, X_train, y_train, X_test, y_actual=None):
        ml_model = self.BayesianRidgeRegressor_ML(X_train, y_train)
        return self.prediction(ml_model, X_test, y_actual)

    def BayesianRidgeRegressor_ML(self, X_train, y_train):
        ML_Alg = linear_model.BayesianRidge()
        ml_model = ML_Alg.fit(X_train, np.ravel(y_train))
        return ml_model

    ################################################################################################
    #  Multi-layer Perceptron Regression  ##########################################################
    def MLPRegressor_run(self, X_train, y_train, X_test, y_actual=None):
        ml_model = self.MLPRegressor_ML(X_train, y_train)
        return self.prediction(ml_model, X_test, y_actual)

    def MLPRegressor_ML(self, X_train, y_train):
        ML_Alg = MLPRegressor(
            hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.001, batch_size='auto',
            learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
            random_state=0, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
            early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        ml_model = ML_Alg.fit(X_train, np.ravel(y_train))
        return ml_model

    ################################################################################################
    def prediction(self, ml_model, X_test, y_actual=None):
        y_predicted = ml_model.predict(X_test)
        return y_predicted, self.show_results(y_predicted, y_actual)

    def run_with_evidence_and_check_prediction(self, dmp, model, data, y_actual):
        predicted = []
        i = 0
        for index, rows in data.iterrows():
            keys = rows.keys()
            sbn = model
            y_name = y_actual.columns[0]
            y_value = y_actual.iloc[i, 0]
            for k in keys:
                sbn += "defineEvidence({}, {});".format(k, rows.get(k))
            sbn += "run(DMP);"
            dmp.run(sbn)

            # check prediction
            # print("================ Prediction results from BN ================")
            # print(dmp.output)
            with open(dmp.output) as json_file:
                net = json.load(json_file)
                for node in net:
                    for obj, contents in node.items():
                        # print('node: ' + obj)
                        mean = None
                        for attr, values in contents.items():
                            # print(" " + attr + ': ' + str(values))
                            if attr == "marginal":
                                mean = values["MU"]

                        if obj == y_name:
                            # print("================================")
                            # print("{} : Predicted {} : Actual {}".format(y_name, mean, y_value))
                            predicted.append(float(mean))

            i += 1

        return predicted

    def ContinuousBNRegressor_run(self, name, X_train, y_train, X_test, y_actual=None, show=True):
        ssbn = self.ContinuousBNRegressor_ML(name, X_train, y_train)
        return self.ContinuousBNRegressor_prediction(name, ssbn, X_test, y_actual, show)

    def ContinuousBNRegressor_ML(self, name, X_train, y_train):
        # print("*** Continuous BN Regressor  ***")
        #############################
        # Make a csv data file
        csv = r'../TestData/{}_for_test.csv'.format(name)
        # csv = "E:/SW-Posco2019/DATAETL/TestData/big_data_1000000.csv"
        output = r'../Output_BN/{}_ssbn.txt'.format(name)

        df_col = pd.concat([X_train, y_train], axis=1)
        df_col.to_csv(csv, index=None)

        #############################
        # Run MEBN learning
        ts = time.time()
        hml = HML_runner()

        # Make a V-BN model
        parents = []
        child = y_train.columns[0]
        for nodeName in X_train.columns:
            parents.append(nodeName)

        hml.make_Model(child, parents)

        # Run MEBN learning
        ssbn = hml.run(csv, output)

        # print("== Mahcine learning end: Time {} ==============================".format(time.time()-ts))
        return ssbn

    def ContinuousBNRegressor_prediction(self, name, ssbn, X_test, y_actual=None, show=True):
        #############################
        # Prediction
        ts = time.time()
        output = r"../Output_BN/{}_bn_output.json".format(name)
        dmp = DMP_runner(output)

        y_predicted = self.run_with_evidence_and_check_prediction(dmp, ssbn, X_test, y_actual)

        # print("== BN was completed : Time {} ==============================".format(time.time()-ts))

        if show == True and len(y_actual) > 1:
            return y_predicted, self.show_results(y_predicted, y_actual)
        else:
            return y_predicted, self.show_results(y_predicted, y_actual=None)




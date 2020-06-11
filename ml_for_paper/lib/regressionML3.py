from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from lightgbm_sklearn import LightGBM
# from deep_learning_regressor import DeepLearningRegressor
from xgboost_sklearn import XGBoost
from ml import ML
import numpy as np
from HML_runner import HML_runner
from DMP_runner import DMP_runner
from sklearn.svm import SVR
import pandas as pd
import numpy as np
import time
import json


class RegressionML(ML):
    def __init__(self):
        super().__init__()
        self.regressors = [
            # RandomForestRegressor(n_estimators=100),
            LinearRegression(),
            # XGBoost(),
            LightGBM(),
            # DeepLearningRegressor(type='custom'),
            # linear_model.Ridge(alpha=3.0),
            # linear_model.BayesianRidge(),
            # DecisionTreeRegressor(),
            # GradientBoostingRegressor(),
            # MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.001, batch_size='auto',
            #     learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
            #     random_state=0, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
            #     early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
            # GaussianProcessRegressor(kernel=DotProduct() + WhiteKernel(), random_state=0),
            # SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
        ]

    def acronym(self, str):
        if str == 'LightGBM':
            return 'LG'
        elif str == 'XGBoost':
            return 'XG'
        elif str == 'RandomForestRegressor':
            return 'RF'
        elif str == 'LinearRegression':
            return 'LR'

    def perform_ML(self, score_function=None):
        # print(f'#==============================================================#')
        # print('X Variables:')
        # print(self.X.columns)
        # print('y Variable:', self.target_variable)
        print(f'Training size[{len(self.y_train)}], Test size[{len(self.y_test)}]')
        # print(f'#==============================================================#')


        scores = {}
        y_preds = {}

        # iterate over classifiers
        for clf in self.regressors:
            name = type(clf).__name__
            info = ''
            if name == 'SVC':
                info = clf.kernel
            if name == 'DeepLearningRegressor':
                info = clf.type

            clf.fit(self.X_train, self.y_train)

            y_pred = clf.predict(self.X_test)

            # print('======================================================')
            # print('len pre', len(y_pred), 'len x_text ', len(self.X_test))
            # print('======================================================')

            if score_function is None:
                score = r2_score(self.y_test, y_pred)
            elif score_function == 'mean_absolute_error':
                score = mean_absolute_error(self.y_test, y_pred)

            # score = clf.score(self.X_test, self.y_test)
            # print('Mean Accuracy Score:')
            # print(f'{name} {info}\t', score)

            scores[name] = []
            scores[name].append(score)

            y_preds[name] = np.ravel(y_pred)

        return scores, y_preds

    #========================================================================================#
    #                                       CLG BN                                           #
    #========================================================================================#
    def run_with_evidence_and_check_prediction(self, dmp, model, data):
        predicted = {}

        for index, rows in data.iterrows():
            keys = rows.keys()
            sbn = model
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
                        predicted[obj] = []
                        for attr, values in contents.items():
                            # print(" " + attr + ': ' + str(values))
                            if attr == "marginal":
                                predicted[obj].append(float(values["MU"]))
                                predicted[obj].append(float(values["SIGMA"]))
                            elif attr == "evidence":
                                predicted[obj].append(float(values["mean"]))
                                predicted[obj].append(0)

        return predicted

    def ContinuousBNRegressor_ML(self, name, X_train, y_train, working_path):
        # print("*** Continuous BN Regressor  ***")
        #############################
        # Make a csv data file
        csv = f'{working_path}/{name}_for_test.csv'
        # csv = "E:/SW-Posco2019/DATAETL/TestData/big_data_1000000.csv"
        output = f'{working_path}/{name}_ssbn.txt'

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

    def ContinuousBNRegressor_prediction(self, name, ssbn, X_test, working_path, show=True):
        #############################
        # Prediction
        ts = time.time()
        output = f'{working_path}/{name}_bn_output.json'
        # output = r"../Output_BN/{}_bn_output.json".format(name)
        dmp = DMP_runner(output)

        return self.run_with_evidence_and_check_prediction(dmp, ssbn, X_test)

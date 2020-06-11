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
# from lightgbm_sklearn import LightGBM
# from deep_learning_regressor import DeepLearningRegressor
# from xgboost_sklearn import XGBoost
from ml import ML
import numpy as np


class RegressionML(ML):
    def __init__(self):
        super().__init__()
        self.regressors = [
            # XGBoost(),
            # LightGBM(),
            # DeepLearningRegressor(type='custom'),
            # linear_model.Ridge(alpha=3.0),
            # linear_model.BayesianRidge(),
            RandomForestRegressor(n_estimators=100),
            # DecisionTreeRegressor(),
            # GradientBoostingRegressor(),
            LinearRegression(),
            # MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.001, batch_size='auto',
            #     learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
            #     random_state=0, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
            #     early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
            # GaussianProcessRegressor(kernel=DotProduct() + WhiteKernel(), random_state=0),
            # SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
        ]

    def perform_ML(self, score_function=None):
        print(f'#==============================================================#')
        print('X Variables:')
        print(self.X.columns)
        print('y Variable:', self.target_variable)
        print(f'Training size[{len(self.y_train)}], Test size[{len(self.y_test)}]')
        print(f'#==============================================================#')
        print('Mean Accuracy Score:')

        scores = {}
        y_tests = {}
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

            if score_function is None:
                score = r2_score(self.y_test, y_pred)
            elif score_function == 'mean_absolute_error':
                score = mean_absolute_error(self.y_test, y_pred)

            # score = clf.score(self.X_test, self.y_test)

            print(f'{name} {info}\t', score)

            scores[name] = []
            scores[name].append(score)

            y_tests[name] = []
            y_tests[name].append(self.y_test)

            y_preds[name] = []
            y_preds[name].append(y_pred)

        return scores, y_tests, y_preds
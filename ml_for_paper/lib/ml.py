from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pandas as pd
import warnings

# For readability, ignore many warnings
warnings.filterwarnings("ignore")


class ML():
    def __init__(self):
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def get_one_hot_data(self, data, categorical_vars):
        refined_data = pd.get_dummies(data, columns=categorical_vars, prefix=categorical_vars)
        return refined_data

    def remove_unnecessary_variables(self, data, unnecessary_vars):
        refined_data = data.copy()

        for c in unnecessary_vars:
            del refined_data[c]

        return refined_data

    def get_numeric_data(self, data, categorical_vars):
        refined_data = data.copy()
        for c in categorical_vars:
            refined_data[c] = refined_data[c].astype('category')
            refined_data[c] = refined_data[c].cat.codes

        return refined_data

    def get_X_y_data(self, data, target_variable):
        y = data[target_variable]
        X = data.drop(target_variable, axis=1)
        y = y.to_frame(target_variable)# <<<<<<
        self.X = X
        self.y = y
        self.target_variable = target_variable
        return X, y

    def set_data(self, X, y):
        # normalize data
        X = StandardScaler().fit_transform(X)

        # split into training and test part
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=.2, random_state=42)

    def set_data2(self, X_train, X_test, y_train, y_test):
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

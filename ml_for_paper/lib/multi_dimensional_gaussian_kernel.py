from numpy import array, linspace
from sklearn.neighbors.kde import KernelDensity
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema

from one_dimensional_gaussian_kernel import OneDimensionalGaussianKernel


class MultiDimensionalGaussianKernel():
    def __init__(self, bandwidth=1):
        self.bandwidth = bandwidth
        self.dict_variables = {}

    def fit(self, X):
        column_len = X.shape[1]
        for i in range(column_len):
            x = X[:, [i]]
            gk = OneDimensionalGaussianKernel(self.bandwidth)
            gk.fit(x)

            self.dict_variables[i] = gk

        for i, gk in self.dict_variables.items():
            print(i, gk)

    def predict(self, X):
        column_len = X.shape[1]
        predicted_list = []
        for i in range(column_len):
            x = X[:, [i]]
            gk = self.dict_variables[i]
            predicted = gk.predict(x)
            predicted_list.append(predicted)

        predicted = list(zip(*predicted_list))
        predicted = [str(x) for x in predicted]

        return predicted

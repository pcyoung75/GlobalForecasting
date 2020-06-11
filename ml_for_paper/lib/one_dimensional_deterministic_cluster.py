from collections import OrderedDict
import pandas as pd
import numpy as np


class OneDimensionalDeterministicCluster():
    def __init__(self, threshold=1000):
        self.threshold = threshold
        self.n_components = 0

    def fit(self, X):
        data_dict = {}
        if isinstance(X, pd.DataFrame):
            for i in range(len(X.index)):
                v = X.iloc[i].values[0]
                if v not in data_dict:
                    data_dict[v] = 1
                else:
                    data_dict[v] += 1
        else:
            for i in range(len(X)):
                # v = np.array([X[i]])
                v = X[i][0]
                if v not in data_dict:
                    data_dict[v] = 1
                else:
                    data_dict[v] += 1

        data_dict = OrderedDict(sorted(data_dict.items()))
        print(data_dict)

        mins = []
        pre_v = float("-inf")
        pre_c = 0
        for i, (v, c) in enumerate(data_dict.items()):

            if i == 0:
                mins.append(v - 10)

            if c > self.threshold or pre_c > self.threshold:
                # make a bin
                mins.append(v - (v - pre_v) / 2)

            pre_v = v
            pre_c = c

        mins.append(pre_v + 10)
        self.mins = mins

    def predict(self, X):
        mins = []
        mins.append(float('-inf'))
        [mins.append(m) for m in self.mins]
        mins.append(float('inf'))

        predicted = []
        for x in X:
            index = -1
            pre_min = None
            for min in mins:
                if pre_min is not None:
                    if pre_min < x and x <= min:
                        break

                pre_min = min
                index += 1

            predicted.append(index)

        self.n_components = max(predicted)+1

        return predicted


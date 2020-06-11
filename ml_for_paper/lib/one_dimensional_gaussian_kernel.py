from numpy import array, linspace
from sklearn.neighbors.kde import KernelDensity
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema


class OneDimensionalGaussianKernel():
    def __init__(self, bandwidth=1):
        self.bandwidth = bandwidth
        self.n_components = 0

    def fit(self, X):
        self.kde = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth).fit(X)
        self.line_min = X.min()
        self.line_max = X.max()
        s = linspace(self.line_min, self.line_max)
        e = self.kde.score_samples(s.reshape(-1, 1))
        self.mins, self.maxs = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]
        self.mins, self.maxs = s[self.mins], s[self.maxs]

        # self.show()

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

    def get_range_info(self, cluster_id):
        if cluster_id == 0:
            s = '-Inf'
            e = self.mins[cluster_id]
        elif cluster_id == (self.n_components-1):
            s = self.mins[cluster_id-1]
            e = '+Inf'
        else:
            s = self.mins[cluster_id-1]
            e = self.mins[cluster_id]

        return f'{s} < x < {e}'

    def show(self):
        s = linspace(self.line_min, self.line_max)
        e = self.kde.score_samples(s.reshape(-1, 1))

        # draw data
        plt.figure(figsize=(5, 5))
        plt.scatter(s, e, marker='o', s=25, edgecolor='k')
        plt.show()

        # find min and max
        mi, ma = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]
        print("Minima:", s[mi])
        print("Maxima:", s[ma])

        plt.plot(s[:mi[0] + 1], e[:mi[0] + 1], 'r',
                 s[mi[0]:mi[1] + 1], e[mi[0]:mi[1] + 1], 'g',
                 s[mi[1]:], e[mi[1]:], 'b',
                 s[ma], e[ma], 'go',
                 s[mi], e[mi], 'ro')

        plt.show()


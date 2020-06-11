import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice


class cluster_test:
    def __init__(self):
        self.datasets = []

    def set_data(self, X_data, y_data):
        new_data = (  # First data set
                        (  # X predictors [X1, X2]
                            # e.g., ) np.array([[10, 20], [20, 30], [11, 11], [11, 24], [25, 36], [12, 11], [30, 20]]),
                            X_data,
                            # Y target [Y]
                            # e.g., ) np.array([1, 0, 1, 1, 1, 0, 1])
                            y_data
                        ),
                        {  # Algorithm Parameters
                            # 'damping': 0.77, 'preference': -240,
                            # 'quantile': 0.2, 'n_clusters': 2, 'min_samples': 20, 'xi': 0.25
                        }
                    )

        self.datasets.append(new_data)

        self.default_base = {'quantile': .3,
                             'eps': .3,
                             'damping': .9,
                             'preference': -200,
                             'n_neighbors': 10,
                             'n_clusters': 5,
                             'min_samples': 10,
                             'xi': 0.05,
                             'min_cluster_size': 0.1}

    def run(self):
        plt.figure(figsize=(9 * 2 + 3, len(self.datasets)*2))
        # plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)

        plot_num = 1

        for i_dataset, (dataset, algo_params) in enumerate(self.datasets):
            # update parameters with dataset-specific values
            params = self.default_base.copy()
            params.update(algo_params)

            X, y = dataset

            # normalize dataset for easier parameter selection
            X = StandardScaler().fit_transform(X)

            # estimate bandwidth for mean shift
            bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])

            # connectivity matrix for structured Ward
            connectivity = kneighbors_graph(X, n_neighbors=params['n_neighbors'], include_self=False)

            # make connectivity symmetric
            connectivity = 0.5 * (connectivity + connectivity.T)

            # ============
            # Create cluster objects
            # ============
            ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
            two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
            ward = cluster.AgglomerativeClustering(n_clusters=params['n_clusters'], linkage='ward', connectivity=connectivity)
            spectral = cluster.SpectralClustering(n_clusters=params['n_clusters'], eigen_solver='arpack', affinity="nearest_neighbors")
            dbscan = cluster.DBSCAN(eps=params['eps'])
            optics = cluster.OPTICS(min_samples=params['min_samples'], xi=params['xi'], min_cluster_size=params['min_cluster_size'])
            affinity_propagation = cluster.AffinityPropagation(damping=params['damping'], preference=params['preference'])
            average_linkage = cluster.AgglomerativeClustering(linkage="average", affinity="cityblock", n_clusters=params['n_clusters'], connectivity=connectivity)
            birch = cluster.Birch(n_clusters=params['n_clusters'])
            gmm = mixture.GaussianMixture(n_components=params['n_clusters'], covariance_type='full')

            clustering_algorithms = (
                ('MiniBatchKMeans', two_means),
                ('AffinityPropagation', affinity_propagation),
                ('MeanShift', ms),
                ('SpectralClustering', spectral),
                ('Ward', ward),
                ('AgglomerativeClustering', average_linkage),
                ('DBSCAN', dbscan),
                ('OPTICS', optics),
                ('Birch', birch),
                ('GaussianMixture', gmm)
            )

            for name, algorithm in clustering_algorithms:
                t0 = time.time()

                # catch warnings related to kneighbors_graph
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="the number of connected components of the " +
                        "connectivity matrix is [0-9]{1,2}" +
                        " > 1. Completing it to avoid stopping the tree early.",
                        category=UserWarning)
                    warnings.filterwarnings(
                        "ignore",
                        message="Graph is not fully connected, spectral embedding" +
                        " may not work as expected.",
                        category=UserWarning)
                    algorithm.fit(X)

                t1 = time.time()
                if hasattr(algorithm, 'labels_'):
                    y_pred = algorithm.labels_.astype(np.int)
                else:
                    y_pred = algorithm.predict(X)

                plt.subplot(len(self.datasets), len(clustering_algorithms), plot_num)
                if i_dataset == 0:
                    plt.title(name, size=9)

                colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                     '#f781bf', '#a65628', '#984ea3',
                                                     '#999999', '#e41a1c', '#dede00']),
                                              int(max(y_pred) + 1))))
                # add black color for outliers (if any)
                colors = np.append(colors, ["#000000"])
                plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

                plt.xlim(-2.5, 2.5)
                plt.ylim(-2.5, 2.5)
                plt.xticks(())
                plt.yticks(())
                plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                         transform=plt.gca().transAxes, size=8,
                         horizontalalignment='right')
                plot_num += 1

        plt.show()


# Test code


#
# X_data = np.array([[667. ],
#  [693.3],
#  [732.9],
#  [658.9],
#  [702.8],
#  [697.2],
#  [658.7],
#  [723.1],
#  [719.5],
#  [687.4],
#  [704.1],
#  [658.8],
#  [667.8],
#  [703.4]])
# Y = np.array([38.36,
#  11.06,
#   8.13,
#  45.23,
#  11.16,
#  11.96,
#  40.27,
#   7.01,
#   7.25,
#  11.28,
#   7.21,
#  40.4 ,
#  32.2 ,
#  11.18])
#
# cl = cluster_test(X_data, Y)
# cl.run()
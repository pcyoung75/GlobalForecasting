from regressionML import RegressionML
import pandas as pd
from sklearn.model_selection import train_test_split
from clustering_alg import Clustering_Alg
import threading
from statistics import mean
import math
import warnings
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from datetime import datetime

class MLModelFamily:
    """
    This class is used as a structure of machine learning family
    """

    def __init__(self, id=0):
        self.id = id
        self.models = {}
        self.models = {}
        self.data = {}

    def add_CL(self, cl_alg):
        self.models[cl_alg] = {}
        self.data[cl_alg] = {}

    def get_CL(self):
        return self.models

    def del_CL(self, cl_alg):
        del self.models[cl_alg]
        del self.data[cl_alg]

    def add_CL_param(self, cl_alg, parameters):
        self.models[cl_alg][parameters] = {}
        self.data[cl_alg][parameters] = {}

    def del_CL_param(self, cl_alg, parameters):
        del self.models[cl_alg][parameters]
        del self.data[cl_alg][parameters]

    def add_CL_model(self, cl_alg, parameters, cl):
        if 'CL_MODEL' not in self.models[cl_alg][parameters]:
            self.models[cl_alg][parameters]['CL_MODEL'] = {}

        self.models[cl_alg][parameters]['CL_MODEL'] = cl

    def set_CL_model_avg_r2(self, cl_alg, parameters, avg_r2):
        self.models[cl_alg][parameters]['AVG_R2'] = avg_r2

    def get_CL_model_avg_r2(self, cl_alg, parameters):
        return self.models[cl_alg][parameters]['AVG_R2']

    def set_CL_total_avg_r2(self, cl_alg, avg_r2):
        self.models[cl_alg]['TOTAL_AVG_R2'] = avg_r2

    def get_CL_total_avg_r2(self, cl_alg):
        return self.models[cl_alg]['TOTAL_AVG_R2']

    def set_CL_data(self, cl_alg, parameters, cur_cluster, data_name, data):
        if cur_cluster not in self.data[cl_alg][parameters]:
            self.data[cl_alg][parameters][cur_cluster] = {}
        self.data[cl_alg][parameters][cur_cluster][data_name] = data

    def add_SL(self, cl_alg, parameters, cur_cluster, prediction_alg):
        if cur_cluster not in self.models[cl_alg][parameters]:
            self.models[cl_alg][parameters][cur_cluster] = {}

        self.models[cl_alg][parameters][cur_cluster][prediction_alg] = {}

    def del_SL(self, cl_alg, parameters, cur_cluster, prediction_alg):
        del self.models[cl_alg][parameters][cur_cluster][prediction_alg]

    def add_SL_model(self, cl_alg, parameters, cur_cluster, prediction_alg, model):
        self.models[cl_alg][parameters][cur_cluster][prediction_alg]['SL_MODEL'] = model

    def set_SL_model_r2(self, cl_alg, parameters, cur_cluster, prediction_alg, r2):
        if math.isnan(r2):
            # In some cases, the training data has the size of 1, then R2 becomes NaN.
            warnings.warn(f'{cl_alg}.{parameters}.{cur_cluster}.{prediction_alg}: The prediction result was NaN.')
            self.models[cl_alg][parameters][cur_cluster][prediction_alg]['R2'] = -10
        else:
            self.models[cl_alg][parameters][cur_cluster][prediction_alg]['R2'] = r2

    def get_SL_model_r2(self, cl_alg, parameters, cur_cluster, prediction_alg):
        return self.models[cl_alg][parameters][cur_cluster][prediction_alg]['R2']

    def get_sl_models(self):
        """
        This returns all selected supervised learning models in the ML model family
        :return: a dictionary for pairs of supervised learning models
        e.g., ) {0: [SL model object], 1: [SL model object], ...}
        0 and 1 stand for the cluster id
        """

        cl = next(iter(self.models.values()))
        cl_model = next(iter(cl.values()))
        result = {}

        for cluster_id, value in cl_model.items():
            if cluster_id is not 'CL_MODEL' and cluster_id is not 'AVG_R2':
                sl_model = next(iter(value.values()))
                result[cluster_id] = sl_model

        return result

    def get_sl_model(self, cluster_id):
        """
        This returns a selected supervised learning model associated with a cluster id
        :param cluster_id: the index of a cluster in a clustering model in the ML model family
        :return: a SL model name, a SL model object
        """

        cl = next(iter(self.models.values()))
        cl_model = next(iter(cl.values()))

        for key, value in cl_model.items():
            if key is not 'CL_MODEL' and key is not 'AVG_R2':
                if key == cluster_id:
                    sl_model = next(iter(value.values()))
                    return list(value.keys())[0], sl_model['SL_MODEL']

        return None, None

    def get_cl_model(self):
        """
        This returns the clustering algorithm class
        """

        cl = next(iter(self.models.values()))
        cl_model = next(iter(cl.values()))
        if isinstance(cl_model['CL_MODEL'], Clustering_Alg):
            return cl_model['CL_MODEL']
        else:
            return None

    def get_clustering_alg(self):
        """
        This returns a selected clustering algorithm
        """

        cl_model = self.get_cl_model()
        if cl_model is not None:
            return cl_model.get_selected_clustering_alg()
        else:
            return None

    def get_CL_models(self, cl_alg, parameters):
        cl_models = {}
        for key, value in self.models[cl_alg][parameters].items():
            if key is not 'CL_MODEL' and key is not 'AVG_R2':
                cl_models[key] = value

        return cl_models

    def set_high_CL_model(self, cl_alg, alg_high):
        self.models[cl_alg]['HIGH_CL'] = alg_high

    def get_high_CL_model(self, cl_alg):
        return self.models[cl_alg]['HIGH_CL']

    def get_clustered_data_by_id(self, cluster_id):
        """
        This returns a clustered data associated with a cluster id
        :param cluster_id: the index of a cluster in a clustering model in the ML model family
        :return: X data, y data
        """

        cl = next(iter(self.data.values()))
        cl_data= next(iter(cl.values()))

        X, y = None, None
        for key, data_dict in cl_data.items():
            if key == cluster_id:
                X = data_dict['x_train']
                y = data_dict['y_train']
                break

        return X, y

    def get_clustered_all_data_by_id(self, cluster_id):
        X, y = self.get_clustered_data_by_id(cluster_id)
        data = pd.concat((X, y), axis=1)
        return data

class DataClusterBasedMachineLearning:
    """
    Data Cluster based Machine Learning (DC-ML)
    """

    def __init__(self, data_x, data_y, clustering_variables, clustering_algs, prediction_algs, metrics='MAE'):
        """
        :param data: A training data set D
        :param clustering_alg: A set of clustering algorithms C
        :param prediction_alg: A set of prediction algorithms P
        :param max_clusters: A maximum number of clusters m
        """
        self.data_x = data_x
        self.data_y = data_y
        self.clustering_variables = clustering_variables
        self.clustering_algs = clustering_algs
        self.prediction_algs = prediction_algs

        # Initialize an ML Model Family
        # It contains all ML models (i.e., Clustering Models (CL) and their Supervised Learning (SL) Models)
        self.ml_family = MLModelFamily()

        # For experiment
        self.is_experiment = True
        self.set_metrics(metrics)

    def set_metrics(self, metrics):
        # Set metrics
        self.metrics = metrics
        if metrics == 'R2':
            self.score = r2_score
        elif metrics == 'MAE':
            self.score = mean_absolute_error

    def min_or_max(self):
        if self.metrics == 'R2':
            return max, float('-inf')
        elif self.metrics == 'MAE':
            return min, float('inf')

    def do_machine_learning(self, prediction_alg, x_train, y_train, cbn_name='temp'):
        """
        This performs common SL learning
        :param prediction_alg: A name of SL learning
        :param x_train: Training data for X variables
        :param y_train: Training data for a y variable
        :param cbn_name: A special parameter for continuous BN learning
        :return: A SL learning object
        """

        if prediction_alg is 'GradientBoostingRegressor':
            model = RegressionML(self.metrics).GradientBoostingRegressor_ML(x_train, y_train)
        elif prediction_alg is 'RandomForestRegressor':
            model = RegressionML(self.metrics).RandomForestRegressor_ML(x_train, y_train)
        elif prediction_alg is 'GaussianProcessRegressor':
            model = RegressionML(self.metrics).GaussianProcessRegressor_ML(x_train, y_train)
        elif prediction_alg is 'LinearRegression':
            model = RegressionML(self.metrics).LinearRegressor_ML(x_train, y_train)
        elif prediction_alg is 'ContinuousBNRegressor':
            model = RegressionML(self.metrics).ContinuousBNRegressor_ML(cbn_name, x_train, y_train)
        return model

    def do_prediction(self, model_name, model, X, y=None, cbn_name='temp'):
        """
        This performs prediction using a given SL learning
        :param model_name: A name of SL learning
        :param model: An object of SL learning
        :param X: Data for X variables
        :param y: Data for a y variable
        :param cbn_name: A special parameter for continuous BN learning
        :return:
        """

        if model_name is 'ContinuousBNRegressor':
            yPredicted, r2 = RegressionML(self.metrics).ContinuousBNRegressor_prediction(cbn_name, model, X, y)
        else:
            yPredicted, r2 = RegressionML(self.metrics).prediction(model, X, y)
        return yPredicted, r2

    def perform_machine_learning_alg(self, cl_alg, parameters, cur_cluster, prediction_alg, x_clustered, y_clustered):
        """
        This performs an SL algorithm
        :param cl_alg: The name of a clustering algorithm
        :param parameters:
        :param cur_cluster: The current cluster index of the clustering algorithm
        :param prediction_alg: A name of SL learning
        :param x_train: Training data for X variables
        :param x_val: Validation data for X variables
        :param y_train: Training data for a y variable
        :param y_val: Validation data for a y variable
        """

        # temporary name for CBN
        cbn_name = f'{cl_alg}_{parameters}_{cur_cluster}_{datetime.now().strftime("%m_%d_%Y %H_%M_%S")}'
        n_splits = 2
        kf = KFold(n_splits=n_splits)

        r2_avg = 0
        for train_index, val_index in kf.split(x_clustered):
            x_train, x_val = x_clustered.iloc[train_index], x_clustered.iloc[val_index]
            y_train, y_val = y_clustered.iloc[train_index], y_clustered.iloc[val_index]

            # perform ML
            model = self.do_machine_learning(prediction_alg, x_train, y_train, cbn_name=cbn_name)

            # perform prediction
            yPredicted, r2 = self.do_prediction(prediction_alg, model, x_val, y_val, cbn_name=cbn_name)
            r2_avg += r2

        r2_avg = r2_avg/n_splits
        # print(cbn_name, prediction_alg, model.feature_importances_)

        self.ml_family.add_SL_model(cl_alg, parameters, cur_cluster, prediction_alg, model)
        self.ml_family.set_SL_model_r2(cl_alg, parameters, cur_cluster, prediction_alg, r2_avg)

    def perform_machine_learning(self, cl_alg, parameters, cur_cluster, x_clustered, y_clustered):
        """
        This executes a set of SL algorithms
        :param cl_alg: The name of a clustering algorithm
        :param parameters:
        :param cur_cluster: The current cluster index of the clustering algorithm
        :param x_train: Training data for X variables
        :param x_val: Validation data for X variables
        :param y_train: Training data for a y variable
        :param y_val: Validation data for a y variable
        """

        #############################################################################
        # 1. perform common machine learning
        thread = []
        for prediction_alg in self.prediction_algs:
            self.ml_family.add_SL(cl_alg, parameters, cur_cluster, prediction_alg)

            print(f'[Thread] {cl_alg}.{parameters}.{cur_cluster} -> ML alg {prediction_alg}')

            t = threading.Thread(target=self.perform_machine_learning_alg, args=([cl_alg, parameters, cur_cluster, prediction_alg, x_clustered, y_clustered]))
            t.setDaemon(True)
            thread.append(t)

        for t in thread:
            t.start()

        for t in thread:
            t.join()

        #############################################################################
        # 2.Select a best prediction alg and remove all low-scored algorithms
        alg_high = None
        min_or_max, r2_high = self.min_or_max()
        for prediction_alg in self.prediction_algs:
            r2 = self.ml_family.get_SL_model_r2(cl_alg, parameters, cur_cluster, prediction_alg)

            print(f'Check R2 for {cl_alg}.{parameters}.{cur_cluster}.{prediction_alg} = {r2}')
            r2_high = min_or_max(r2_high, r2)
            if r2 is r2_high:
                if alg_high is not None:
                    self.ml_family.del_SL(cl_alg, parameters, cur_cluster, alg_high)
                alg_high = prediction_alg
            else:
                self.ml_family.del_SL(cl_alg, parameters, cur_cluster, prediction_alg)

    def perform_clustering_alg_with_clusters(self, cl_alg, parameters):
        """
        This performs a clustering algorithm using a given maximum number of clusters
        :param cl_alg: A clustering algorithm
        :param parameters:
        """

        #############################################################################
        # 1. Perform clustering algorithm using input training data
        print(f'perform_prediction_alg {cl_alg} with the cluster {parameters}')
        cl = Clustering_Alg()
        cl.set_algs(cl_alg)
        cl.set_base(parameters)
        cl.set_data(self.data_x, self.data_y, self.clustering_variables)
        cl.run()

        # Note that the clustering algorithm can change the number of clusters
        # e.g., ) The default n_clusters = 3 changes to n_clusters = 2 according to the clustering result
        self.ml_family.add_CL_model(cl_alg, parameters, cl)

        # Get clustered data from the cluster model
        data, data_x, data_y = cl.get_clustered_data_XY(cl_alg)

        #############################################################################
        # 2. Check cluster consistency:
        # If a cluster contains only one datum, data grouped by the cluster is determined as inconsistent data
        # Then, return it with a lowest score
        for cur_cluster, datum in data_x.items():
            if len(data_y[cur_cluster]) == 1:
                print(len(data_y[cur_cluster]))
            print(f'data split for {cl_alg}.{parameters}.{cur_cluster} X-Size[{len(data_x[cur_cluster])}] Y-Size[{len(data_y[cur_cluster])}]')

            # inconsistent clustered data!
            if len(data_x[cur_cluster]) != len(data_y[cur_cluster]):
                warnings.warn('inconsistent clustered data!')
                self.ml_family.set_CL_model_avg_r2(cl_alg, parameters, -10000)
                return

        #############################################################################
        # 3. Preparing SL learning
        thread = []

        for cur_cluster, datum in data_x.items():
            # split data for machine learning
            # x_train, x_val, y_train, y_val = train_test_split(data_x[cur_cluster], data_y[cur_cluster], test_size=test_size)

            if self.is_experiment:
                # print(f'sub-training data size[{len(y_train)}], validation data size[{len(y_val)}]')

                self.ml_family.set_CL_data(cl_alg, parameters, cur_cluster, 'x_train', data_x[cur_cluster])
                self.ml_family.set_CL_data(cl_alg, parameters, cur_cluster, 'y_train', data_y[cur_cluster])

            t = threading.Thread(target=self.perform_machine_learning, args=([cl_alg, parameters, cur_cluster, data_x[cur_cluster], data_y[cur_cluster]]))
            t.setDaemon(True)
            thread.append(t)

        for t in thread:
            t.start()

        for t in thread:
            t.join()

        #############################################################################
        # 4. Calculate the average R2 and store it to 'ml_models.cl_alg.parameters.avg_r2'
        avg_r2 = []
        cl_models = self.ml_family.get_CL_models(cl_alg, parameters)

        for cur_cluster, ml_alg_r2 in cl_models.items():
            try:
                r2 = list(ml_alg_r2.values())[0]['R2']
            except IndexError:
                print('list index out of range')
            avg_r2.append(r2)

        mean_avg_r2 = mean(avg_r2)
        self.ml_family.set_CL_model_avg_r2(cl_alg, parameters, mean_avg_r2)

    def perform_clustering(self, cl_alg):
        """
        This performs clustering
        :param cl_alg: A clustering algorithm
        """

        #############################################################################
        # 1. Perform clustering
        thread = []
        # for parameters in range(2, self.max_clusters + 1):
        for parameters in self.clustering_algs[cl_alg]:
            self.ml_family.add_CL_param(cl_alg, parameters)
            print(f'[Thread] clustering alg {cl_alg} with the cluster parameter {parameters} start')
            t = threading.Thread(target=self.perform_clustering_alg_with_clusters, args=([cl_alg, parameters]))
            t.setDaemon(True)
            thread.append(t)

        for t in thread:
            t.start()

        for t in thread:
            t.join()

        #############################################################################
        # 2. select a best number of clusters and remove all low-scored models
        cl_num_high = None
        min_or_max, avg_r2_high = self.min_or_max()

        # for parameters in range(2, self.max_clusters + 1):
        for parameters in self.clustering_algs[cl_alg]:
            avg_r2 = self.ml_family.get_CL_model_avg_r2(cl_alg, parameters)

            print(f'Check avg_r2 for {cl_alg}.{parameters} = {avg_r2}')
            avg_r2_high = min_or_max(avg_r2_high, avg_r2)
            if avg_r2 is avg_r2_high:
                if cl_num_high is not None:
                    self.ml_family.del_CL_param(cl_alg, cl_num_high)
                cl_num_high = parameters
            else:
                self.ml_family.del_CL_param(cl_alg, parameters)

        self.ml_family.set_CL_total_avg_r2(cl_alg, avg_r2_high)

    def run(self):
        """
        This is a main function for running DC-ML learning
        :return: An ML model family object
        """

        #############################################################################
        # 1. Perform clustering
        thread = []
        for cl_alg in self.clustering_algs:
            self.ml_family.add_CL(cl_alg)
            print(f'[Thread] {cl_alg} start')
            t = threading.Thread(target=self.perform_clustering, args=([cl_alg]))
            t.setDaemon(True)
            thread.append(t)

        for t in thread:
            t.start()

        for t in thread:
            t.join()

        #############################################################################
        # 2. perform ML for non-cluster models
        # 'NON-CLUSTER' is used for machine learning of SL models without clustering
        # self.ml_family.add_CL('NON-CLUSTER')
        # self.ml_family.add_CL_param('NON-CLUSTER', 1)
        # self.ml_family.add_CL_model('NON-CLUSTER', 1, 0)
        # self.ml_family.set_CL_data('NON-CLUSTER', 1, 0, 'x_train', self.data_x)
        # self.ml_family.set_CL_data('NON-CLUSTER', 1, 0, 'y_train', self.data_y)
        # self.perform_machine_learning('NON-CLUSTER', 1, 0, self.data_x, self.data_y)
        # cl_models = self.ml_family.get_CL_models('NON-CLUSTER', 1)
        # for cur_cluster, ml_alg_r2 in cl_models.items():
        #     r2 = list(ml_alg_r2.values())[0]['R2']
        # self.ml_family.set_CL_total_avg_r2('NON-CLUSTER', r2)

        #############################################################################
        # 3. select a high-scored model
        min_or_max, avg_r2_high = self.min_or_max()
        high_scored_cl = None
        cl_algs = list(self.ml_family.get_CL().keys())
        for cl_alg in cl_algs:
            avg_r2 = self.ml_family.get_CL_total_avg_r2(cl_alg)
            avg_r2_high = min_or_max(avg_r2_high, avg_r2)
            if avg_r2 is avg_r2_high:
                if high_scored_cl is not None:
                    self.ml_family.del_CL(high_scored_cl)

                high_scored_cl = cl_alg
            else:
                self.ml_family.del_CL(cl_alg)

        #############################################################################
        # 4. perform ML **again** using all data of both training and validation data sets
        sl_models = self.ml_family.get_sl_models()
        for cluster_id, sl in sl_models.items():
            # print(cluster_id, sl)
            x_train, y_train = self.ml_family.get_clustered_data_by_id(cluster_id)
            # self.data[cl_alg][parameters]

            # temporary name for CBN
            cbn_name = f'{cl_alg}_{cluster_id}_{datetime.now().strftime("%m_%d_%Y %H_%M_%S")}'

            if isinstance(sl['SL_MODEL'], str):
                sl_name = 'ContinuousBNRegressor'
            else:
                sl_name = type(sl['SL_MODEL']).__name__

            # perform ML
            model = self.do_machine_learning(sl_name, x_train, y_train, cbn_name=cbn_name)

            # Replace the old with the new SL model
            sl['SL_MODEL'] = model

        print('!!! An ML model family was selected !!!')

        return self.ml_family

    def perform_prediction(self, x_test, y_test, metrics='R2'):
        """
        This performs prediction given a test data set
        :param x_test: test data for X variables
        :param y_test: test data for a y variable
        :return: predicted y values, R2 scores
        """

        self.set_metrics(metrics)

        cl_alg = self.ml_family.get_clustering_alg()
        y_label = []

        # None-cl_model means that the Non-Cluster model was selected
        if cl_alg is not None:
            # Get data using clustering variables
            data_for_clustering = x_test[self.clustering_variables]

            cl_model = self.ml_family.get_cl_model()
            # normalize dataset for easier parameter selection
            x_test_norm = cl_model.data_scaler(data_for_clustering)
            # predict label using the normalized data
            y_label = cl_alg.predict(x_test_norm)

            index = 0
            yPredicted = []

            for cluster_id in y_label:
                ml_name, ml_model = self.ml_family.get_sl_model(cluster_id)
                yPre, r2 = self.do_prediction(ml_name, ml_model, x_test.iloc[[index]], y_test.iloc[[index]], cbn_name = datetime.now().strftime("%m_%d_%Y %H_%M_%S"))
                yPredicted.append(yPre)
                index += 1

            yPredicted2 = pd.DataFrame(yPredicted)
            r2 = self.score(y_test, yPredicted2)
        else:
            ml_name, ml_model = self.ml_family.get_sl_model(0)
            yPredicted, r2 = self.do_prediction(ml_name, ml_model, x_test, y_test, cbn_name=datetime.now().strftime("%m_%d_%Y %H_%M_%S"))

        return yPredicted, r2, y_label

import io
import math
import pickle
import random
from sys import modules

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from ipywidgets import HBox, IntSlider, ToggleButtons, interactive_output
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from sklearn.cluster import OPTICS, SpectralClustering
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

from utilities.dummy_utilizer import DummyUtilizer
from utilities.kmedoids import kMedoids
from utilities.lionforests_utility import path_similarity, roundup


class LionForests:
    """LionForests locally interpreting random forests models"""

    def __init__(self, model=None, trained=False, utilizer=None, feature_names=None, class_names=None, categorical_features=None):
        """Init function
        Args:
            model: The trained RF model
            trained: Binary variable to state if there is a trained model
            utilizer: The preferred scaler
            feature_names: The names of the features from our dataset
            class_names: The names of the classes from our dataset
            categorical_features: The categorical features from our dataset
        Attributes:
            model: The classifier/regression model
            utilizer: The scaler, if any
            trees: The trees of an trained ensemble system
            feature_names: The names of the features
            class_names: The names of the two classes
            categorical_features: The categorical features from our dataset
            is_categorical: Maps the categories of the categporical features to the feature names
            numerical_feature_names: Isolates the names of the numerical features
            accuracy: The accuracy of the model (accuracy for classification, mse for regression):
            min_max_feature_values: A helping dictionary for the path/feature reduction process
            number_of_estimators: The amount of trees
            ranked_features: The features ranked based on SHAP Values (Small-Medium Datasets) or Feature Importance (Huge Datasets)
            distribution_plots: Prepared distributions for each feature in order to be used in the visualisation
            mode: 'C' or 'R' based on the class names, in order to know the task
            quorum: Minimum required number of paths for each interpretation to have conclusive rules, only in binary and multi-class classification
            error: The allowed error the user defines to produce the interpretation
        """
        self.model = model
        self.utilizer = utilizer
        if utilizer is not None:  # load scaling
            self.utilizer = utilizer
        elif model is not None and utilizer is None:
            self.utilizer = DummyUtilizer()
        else:
            self.utilizer = MinMaxScaler(feature_range=(0, 1))

        self.trees = None
        if model is not None:
            self.trees = model.estimators_
        self.feature_names = feature_names
        self.categorical_features = categorical_features
        if categorical_features is not None:
            self.is_categorical = {}
            for f in range(len((feature_names))):
                self.is_categorical[feature_names[f]] = 0
                for cat in categorical_features:
                    if cat in feature_names[f]:
                        self.is_categorical[feature_names[f]] = 1
            self.numerical_feature_names = []
            for f, p in self.is_categorical.items():
                if p == 0:
                    self.numerical_feature_names.append(f)

        self.class_names = class_names
        self.accuracy = 0
        self.min_max_feature_values = {}
        self.number_of_estimators = 0
        self.ranked_features = {}
        self.distribution_plots = {}
        if len(self.class_names) >= 2:  # NEWLINE
            self.quorum = 0
            self.mode = 'C'
        else:  # regression
            self.error = 0
            self.mode = 'R'
        if trained:
            print(
                "Please call 'trained(train_data)' in order to collect few statistics of the data distribution")

    def _prepare_distribution_plots(self, train_data):
        """This function prepares, combines and saves the distribution plots of each numerical feature. this will help the response and the portability of an LF instance, without storing the whole dataset
        Args:
            train_data: Training data used in the RF model
        """
        def dist_plot(x, i):
            fig = plt.figure(num=None, figsize=(10, 7),
                             dpi=100, facecolor='w', edgecolor='k')
            plt.hist(x[:, i:i+1], bins='auto')  # 'sqrt')
            plt.close(fig)
            return fig
        if type(train_data) is not np.ndarray:
            data = np.array(train_data)
        else:
            data = train_data
        if self.categorical_features is not None:
            for i in range(len(self.feature_names)):
                if self.is_categorical[self.feature_names[i]] == 0:
                    self.distribution_plots[i] = dist_plot(data, i)
        else:
            for i in range(len(self.feature_names)):
                self.distribution_plots[i] = dist_plot(data, i)

    def fit_trained(self, train_data, train_target):
        """This function prepares computes necessary statistics need LF to work, without maintaining tha training data
        Args:
            train_data: Training data used in the RF model
        Attributes:
            trees: The trees of an trained ensemble system
            min_max_leaf_prediction_per_trees: Traverses the trees to find the min and max predictions in RFs for regression
            number_of_estimators: The amount of trees
            quorum: Minimum required number of paths for each interpretation to have conclusive rules, only in binary and multi-class classification
            error: The allowed error the user defines to produce the interpretation
            min_max_feature_values: A helping dictionary for the path/feature reduction process
            ranked_features: The features ranked based on SHAP Values (Small-Medium Datasets) or Feature Importance (Huge Datasets)
        """
        self._prepare_distribution_plots(train_data)

        train_data = self.utilizer.transform(train_data)

        self.trees = self.model.estimators_

        if self.mode == 'R':
            from utilities.lionforests_utility import \
                find_regression_trees_min_maxes
            self.min_max_leaf_prediction_per_trees = find_regression_trees_min_maxes(
                self.trees, self.feature_names)

        self.number_of_estimators = self.model.n_estimators
        if self.mode == 'R':
            self.error = self.accuracy  # NEWLINE
        else:
            self.quorum = int(self.number_of_estimators / 2 + 1)
        for i in range(len(self.feature_names)):
            self.min_max_feature_values[self.feature_names[i]] = [
                min(train_data[:, i]), max(train_data[:, i])]
        for ind in range(len(self.class_names)):
            d = {'Feature': self.feature_names,
                 'Importance': self.model.feature_importances_}
            self.ranked_features[self.class_names[ind]] = \
                pd.DataFrame(data=d).sort_values(
                    by=['Importance'], ascending=False)['Feature'].values

    def transform_categorical_data(self, train_data, train_target, feature_names):
        """Not ready function
        Args:
            model: The trained RF model
            trained: Binary variable to state if there is a trained model
            utilizer: The preferred scaler
            feature_names: The names of the features from our dataset
            class_names: The names of the classes from our dataset
            categorical_features: The categorical features from our dataset
        Attributes:
            model: The classifier/regression model
            utilizer: The scaler, if any
            trees: The trees of an trained ensemble system
            feature_names: The names of the features
            class_names: The names of the two classes
            categorical_features: The categorical features from our dataset
            is_categorical: Maps the categories of the categporical features to the feature names
            numerical_feature_names: Isolates the names of the numerical features
            accuracy: The accuracy of the model (accuracy for classification, mse for regression):
            min_max_feature_values: A helping dictionary for the path/feature reduction process
            number_of_estimators: The amount of trees
            ranked_features: The features ranked based on SHAP Values (Small-Medium Datasets) or Feature Importance (Huge Datasets)
            distribution_plots: Prepared distributions for each feature in order to be used in the visualisation
            mode: 'C' or 'R' based on the class names, in order to know the task
            quorum: Minimum required number of paths for each interpretation to have conclusive rules, only in binary and multi-class classification
            error: The allowed error the user defines to produce the interpretation
        """
        self.onehot_data = to_onehot_data
        self.onehot_transformation = True
        self.numerical_data = [
            i for i in feature_names if i not in to_onehot_data]

        numerical = pd.DataFrame(train_data, columns=feature_names)[
            self.numerical_data].values
        num = pd.DataFrame(numerical, columns=self.numerical_data)
        if self.onehot_transformation:
            onehot_encoded = pd.DataFrame(train_data, columns=feature_names)[
                to_onehot_data].values
            enc = OneHotEncoder(handle_unknown='ignore')
            enc.fit(onehot_encoded)
            self.onehot_dictionary = {}
            for i in range(len(to_onehot_data)):
                self.onehot_dictionary[str('x' + str(i))] = to_onehot_data[i]
            self.onehot_features = enc.get_feature_names()
            one = pd.DataFrame(enc.transform(onehot_encoded).A,
                               columns=self.onehot_features)

        if self.ordinal_transformation:
            ordinal_encoded = pd.DataFrame(train_data, columns=feature_names)[
                to_ordinal_data].values
            enc = OrdinalEncoder()
            enc.fit(ordinal_encoded)
            self.ordinal_features = to_ordinal_data
            ore = pd.DataFrame(enc.transform(
                ordinal_encoded), columns=to_ordinal_data)

        if self.onehot_transformation and self.ordinal_transformation:
            all_features = []
            for f in self.onehot_features:
                all_features.append(f)
            for f in self.ordinal_features:
                all_features.append(f)
            for f in self.numerical_data:
                all_features.append(f)
            return pd.concat([one, ore, num], axis=1), train_target, all_features
        elif self.onehot_transformation:
            all_features = []
            for f in self.onehot_features:
                all_features.append(f)
            for f in self.numerical_data:
                all_features.append(f)
            return pd.concat([one, num], axis=1), train_target, all_features
        elif self.ordinal_transformation:
            all_features = []
            for f in self.ordinal_features:
                all_features.append(f)
            for f in self.numerical_data:
                all_features.append(f)
            return pd.concat([ore, num], axis=1), train_target, all_features

    def fit(self, train_data, train_target, params=None):
        """ train function is used to train an RF model and extract information like accuracy, model, trees and
        min_max_feature_values among all trees
        Args:
            train_data: The data we are going to use to train the random forest
            train_target: The targets of our train data
            scaling_method: The preffered scaling method. The deafult is MinMaxScaler with feature range -1 to 1
            feature_names: The names of the features from our dataset
            params: The parameters for our gridSearchCV to select the best RF model
        """
        self._prepare_distribution_plots(
            train_data)  # Before scaling because we are going to inverse the explanations in the end.
        self.utilizer.fit(train_data)
        train_data = self.utilizer.transform(train_data)
        if self.mode == 'R':
            random_forest = RandomForestRegressor(
                random_state=0, n_jobs=-1)  # NEWLINE
        else:
            random_forest = RandomForestClassifier(random_state=0, n_jobs=-1)

        parameters = params
        if parameters is None:
            parameters = [{
                'max_depth': [1, 5, 7, 10],  # [1, 5, 7, 10],#1, 5, 7, 10
                # ['sqrt', 'log2', 0.75, None], #'sqrt', 'log2', 0.75, None
                'max_features': ['sqrt', 'log2', 0.75, None],
                'bootstrap': [True, False],  # [True, False], #True, False
                # [1, 2, 5, 10, 0.10], #1, 2, 5, 10, 0.10
                'min_samples_leaf': [1, 2, 5, 10, 0.10],
                # [10, 100, 500, 1000] #10, 100, 500, 1000
                'n_estimators': [10, 100, 500, 1000]
            }]
        if self.mode == 'R':
            clf = GridSearchCV(estimator=random_forest, param_grid=parameters, cv=10,
                               n_jobs=-1, verbose=0, scoring='neg_mean_absolute_error')  # NEWLINE
        else:
            clf = GridSearchCV(estimator=random_forest, param_grid=parameters,
                               cv=10, n_jobs=-1, verbose=0, scoring='f1_weighted')
        clf.fit(train_data, train_target)
        self.accuracy = clf.best_score_
        self.model = clf.best_estimator_
        self.trees = self.model.estimators_
        if self.mode == 'R':
            from utilities.lionforests_utility import \
                find_regression_trees_min_maxes
            self.min_max_leaf_prediction_per_trees = find_regression_trees_min_maxes(
                self.trees, self.feature_names)
        self.number_of_estimators = self.model.n_estimators
        if self.mode == 'R':
            self.error = self.accuracy  # NEWLINE
        else:
            self.quorum = int(self.number_of_estimators / 2 + 1)
        for i in range(len(self.feature_names)):
            self.min_max_feature_values[self.feature_names[i]] = [
                min(train_data[:, i]), max(train_data[:, i])]
        for ind in range(len(self.class_names)):
            d = {'Feature': self.feature_names,
                 'Importance': self.model.feature_importances_}
            self.ranked_features[self.class_names[ind]] = \
                pd.DataFrame(data=d).sort_values(
                    by=['Importance'], ascending=False)['Feature'].values

    def _paths_of_label_(self, instance, prediction, mode):
        ranges = []
        predictions = []
        for tree in self.trees:
            tree_prediction = int(tree.predict([instance])[0])
            if mode == 'R':
                tree_prediction = tree.predict([instance])[0]
            if mode == 'R' or tree_prediction == prediction:
                path = tree.decision_path([instance])
                if mode != 'R':
                    tree_prediction = tree.predict_proba([instance])[
                        0][prediction]
                leq = {}  # leq: less equal ex: x <= 1
                b = {}  # b: bigger ex: x > 0.6
                local_range = {}
                for node in path.indices:
                    feature_id = tree.tree_.feature[node]
                    feature = self.feature_names[feature_id]
                    threshold = tree.tree_.threshold[node]
                    if threshold != -2.0:
                        if instance[feature_id] <= threshold:
                            leq.setdefault(feature, []).append(threshold)
                        else:
                            b.setdefault(feature, []).append(threshold)
                for k in leq:
                    local_range.setdefault(k, []).append(
                        ['<=', min(leq[k])])  # !!
                for k in b:
                    local_range.setdefault(k, []).append(
                        ['>', max(b[k])])  # !!
                ranges.append(local_range)
                predictions.append(tree_prediction)
        return ranges, predictions

    def _path_finder(self, instance, mode=None):
        """_path_finder function finds
        Args:
            instance: The instance we want to find the paths
        Return:
            a list which contains a dictionary with features as keys and their min max ranges as values, as well as the
            number of the paths
        """
        if self.utilizer is not None:
            instance = self.utilizer.transform([instance])[0]
        prediction = int(self.model.predict([instance])[0])
        ranges, probabilities = self._paths_of_label_(
            instance, prediction, mode)
        if len(self.class_names) > 2:

            predictions = self.model.predict_proba([instance])[0]
            voting_scheme = [list(predictions).index(x)
                             for x in sorted(predictions, reverse=True)]
            if len(ranges) >= self.quorum and np.average(np.array(probabilities)) > 0.5:
                other_paths = []
                for k in voting_scheme[1:]:
                    additional_ranges, _ = self._paths_of_label_(
                        instance, k, mode)
                    other_paths = other_paths + additional_ranges
                return ranges, probabilities, other_paths
            else:
                additional_ranges, _ = self._paths_of_label_(
                    instance, voting_scheme[1], mode)  # 0 was the ramges
                mcc = len(additional_ranges) - len(ranges) + \
                    self.number_of_estimators + 1
                ranges = ranges + additional_ranges
                if len(ranges) == self.number_of_estimators:
                    other_paths = []
                    for k in voting_scheme[2:]:
                        additional_ranges, _ = self._paths_of_label_(
                            instance, k, mode)
                        other_paths = other_paths + additional_ranges
                    return ranges, other_paths

                if mcc >= self.number_of_estimators:
                    for k in voting_scheme[2:]:
                        additional_ranges, _ = self._paths_of_label_(
                            instance, k, mode)
                        ranges = ranges + additional_ranges
                    return ranges, []
                else:
                    extra = mcc - len(ranges)
                    other_paths = []
                    for k in voting_scheme[2:]:
                        additional_ranges, _ = self._paths_of_label_(
                            instance, k, mode)
                        other_paths = other_paths + additional_ranges
                    ranges = ranges + other_paths[:extra]
                    other_paths = other_paths[extra:]
                    return ranges, other_paths
        return ranges, probabilities

    def explain(self, instance, reduction=True, ar_algorithm=None, cl_algorithm=None, save_plots=False, instance_qe=0, clusters=0, method='123', to_vis=False):
        """Explain function finds a single range rule which will be the explanation for the prediction
        of this instance
        Args:
            instance: The instance we want to find the paths
            reduction: The targets of our train data
            save_plots: The bar and stacked area plots for every feature will be saved
            complexity:
        Return:
            a feature range rule which will be the explanation
        """
        original_instance = instance.copy()

        # It will fill in only if AR reduction will be applied!
        self.silly_local_importance = {}
        # Notifying user about the reduction algorithms
        if reduction:
            if self.mode == 'R':
                if method != 'R1' and method != 'R2' and method != 'R3':
                    print("LionForests on regression tasks can reduce features and paths through only 3 techniques. 1) Association Rules + Random Selection (R1), 2) Inner Distribution (R2) and 3) Outter Ddistribution (R3)")
                    print('Setting the default: R1')
                    method = 'R1'
            if ar_algorithm is None:
                ar_algorithm = 'fpgrowth'
            if ar_algorithm is not None and ar_algorithm not in ['apriori', 'fpgrowth']:
                print(
                    "The algorithm you requested is not available. I can currently use: Apriori and FP-Growth.")
                print("I will use FP-Growth instead!")
                ar_algorithm = 'fpgrowth'

            if cl_algorithm is None:
                cl_algorithm = 'kmedoids'
            if cl_algorithm is not None and cl_algorithm not in ['kmedoids', 'OPTICS', 'SC']:
                print(
                    "The algorithm you requested is not available. I can currently use: kmedoids, OPTICS and SC.")
                print("I will use kmedoids instead!")
                cl_algorithm = 'kmedoids'

        # Setting the quorum. Normally should be half plus one path of all paths.
        # For eg. in 100 trees the quorum is 51. However, a user can set the quorum to a higher
        # number, quorum = 90 trees.
        local_error = None
        if instance_qe <= 0:
            if self.mode == 'R':
                instance_qe = abs(self.error)
            else:
                instance_qe = self.quorum

        number_of_clusters = clusters
        if clusters <= 0:
            number_of_clusters = 5
            if self.number_of_estimators < 5:
                number_of_clusters = self.number_of_estimators
            if self.number_of_estimators >= 100:
                number_of_clusters = int(
                    math.ceil(instance_qe * 3 / 22))  # 1100 = 11 * 100

        # Finding the original rules
        reducability = False  # It will be used only for multiclass problems
        if len(self.class_names) > 2:
            results = self._path_finder(instance)
            if len(results) == 3:
                must_rules = results[0]
                must_probabilities = results[1]
                additional_rules = results[2]
                reducability = True
            else:
                must_rules = results[0]
                additional_rules = results[1]
                reducability = False
            rules = must_rules + additional_rules
        elif len(self.class_names) == 2:
            rules, probabilities = self._path_finder(instance)
        else:  # NEWLINE REGRESSION
            rules, predictions = self._path_finder(instance, 'R')
        # Finding the number of original rules
        original_number_of_rules = len(rules)

        # Finding the features needes for the original rules
        items = set()
        for pr in rules:
            for p in pr:
                items.add(p)
        local_feature_names = list(items)

        if to_vis:
            original_feature_rule_limits = {}
            for feature in self.feature_names:
                if feature in local_feature_names:
                    mi, ma = self._pre_feature_range_caluclation(
                        rules, feature)
                    original_feature_rule_limits[feature] = [self.utilizer.inverse_transform(np.array([mi * np.ones(len(self.feature_names)), mi * np.ones(len(self.feature_names))]))[0][self.feature_names.index(feature)],
                                                             self.utilizer.inverse_transform(np.array([ma * np.ones(len(self.feature_names)), ma * np.ones(len(self.feature_names))]))[0][self.feature_names.index(feature)]]

        # Finding the number of original features
        original_number_of_features = len(local_feature_names)
        original_features = local_feature_names.copy()

        # If reduction was requested and there are trees to reduce then call _reduce_rules method
        if reduction and instance_qe < original_number_of_rules:
            if len(self.class_names) > 2:
                if reducability:
                    temp_rules = self._reduce_rules(
                        must_rules, must_probabilities, instance_qe, number_of_clusters, ar_algorithm, cl_algorithm, method, save_plots, None)
                    if len(temp_rules[0]) != 0:
                        rules = temp_rules[0]
                        local_feature_names = temp_rules[1]
                else:
                    rules = must_rules
                    items = set()
                    for pr in rules:
                        for p in pr:
                            items.add(p)
                    local_feature_names = list(items)
            elif len(self.class_names) == 2:
                temp_rules = self._reduce_rules(
                    rules, probabilities, instance_qe, number_of_clusters, ar_algorithm, cl_algorithm, method, save_plots, None)
                if len(temp_rules[0]) != 0:
                    rules = temp_rules[0]
                    local_feature_names = temp_rules[1]
            # in regression we use only association rules and random selection extended, integrated inside association rules (method == 1)
            else:
                temp_rules = self._reduce_rules(
                    rules, predictions, instance_qe, number_of_clusters, ar_algorithm, cl_algorithm, method, save_plots, instance)
                if len(temp_rules[0]) != 0:
                    rules = temp_rules[0]
                    local_feature_names = temp_rules[1]
                    local_error = temp_rules[2]
        # Let's build our final rule
        rule = "if "
        temp_f_mins = {}
        temp_f_maxs = {}
        feature_rule_limits = {}
        for feature in self.feature_names:
            if feature in local_feature_names:
                if save_plots:
                    bars, _ = self._pre_feature_range_caluclation_bars(
                        rules, feature)
                    if bars != False:
                        # aggregation = self._aggregated_feature_range(bars, feature, save_plots) #Only for the experiments
                        aggregation = self._aggregated_feature_range(
                            bars, feature, False)
                        temp_f_mins[feature] = aggregation[0]
                        temp_f_maxs[feature] = aggregation[1]
                else:
                    mi, ma = self._pre_feature_range_caluclation(
                        rules, feature)
                    temp_f_mins[feature] = mi
                    temp_f_maxs[feature] = ma

        f_mins = []
        f_maxs = []
        for feature in self.feature_names:
            if feature in temp_f_mins:
                f_mins.append(temp_f_mins[feature])
            else:
                f_mins.append(0)
            if feature in temp_f_maxs:
                f_maxs.append(temp_f_maxs[feature])
            else:
                f_maxs.append(0)
        if self.utilizer is not None:
            instance = self.utilizer.transform([instance])[0]

        if self.mode == 'R':
            if local_error is not None:
                decision = self.class_names[0] + ': ' + str(round(self.model.predict(
                    [instance])[0], 4)) + " +/- " + str(round(local_error, 4)) + " error"
            else:
                decision = self.class_names[0] + ': ' + str(round(self.model.predict(
                    [instance])[0], 4))
            class_name = self.class_names[0]
        else:
            class_name = self.class_names[int(
                self.model.predict([instance])[0])]
            decision = class_name

        if self.categorical_features is not None:
            original_features_cat = 0
            for feat in original_features:
                if self.is_categorical[feat] == 0:
                    original_features_cat = original_features_cat + 1
                if self.is_categorical[feat] == 1:
                    index_f = self.feature_names.index(feat)
                    if instance[index_f] == 1:
                        original_features_cat = original_features_cat + 1
            local_features_cat = 0
            for feat in local_feature_names:
                if self.is_categorical[feat] == 0:
                    local_features_cat = local_features_cat + 1
                if self.is_categorical[feat] == 1:
                    index_f = self.feature_names.index(feat)
                    if instance[index_f] == 1:
                        local_features_cat = local_features_cat + 1

            categorical_features_alternatives = {}
            for ranked_f in self.ranked_features[class_name]:
                f = self.feature_names.index(ranked_f)
                if self.is_categorical[ranked_f] == 1:
                    if instance[f] == 1:
                        if ranked_f not in local_feature_names:  # This means that the categorical feature is reduced so we need to show to the user the alternative options
                            categorical_features_alternatives[ranked_f] = []
                            for cat in self.categorical_features:
                                if cat in ranked_f:
                                    for k in local_feature_names:
                                        if cat in k:
                                            categorical_features_alternatives[ranked_f].append(
                                                k)
                        else:  # Else add this caterical feature to the rule
                            for ff in self.categorical_features:
                                if ff in self.feature_names[f]:
                                    if self.feature_names[f] in local_feature_names:
                                        if self.utilizer is not None:
                                            mmi = self.utilizer.inverse_transform(
                                                np.array([f_mins, f_mins]))[0][f]
                                            mma = self.utilizer.inverse_transform(
                                                np.array([f_maxs, f_maxs]))[0][f]
                                        else:
                                            mmi = np.array(
                                                [f_mins, f_mins])[0][f]
                                            mma = np.array(
                                                [f_maxs, f_maxs])[0][f]
                                        if str(round(mma, 3)) == '1.0':
                                            feature_rule_limits[self.feature_names[f]] = [
                                                mmi, mma]
                                            rule = rule + \
                                                self.feature_names[f] + " & "
                else:
                    if self.feature_names[f] in local_feature_names:
                        if self.utilizer is not None:
                            mmi = self.utilizer.inverse_transform(
                                np.array([f_mins, f_mins]))[0][f]
                            mma = self.utilizer.inverse_transform(
                                np.array([f_maxs, f_maxs]))[0][f]
                        else:
                            mmi = np.array([f_mins, f_mins])[0][f]
                            mma = np.array([f_maxs, f_maxs])[0][f]
                        feature_rule_limits[self.feature_names[f]] = [mmi, mma]
                        rule = rule + \
                            str(round(mmi, 3)) + "<=" + \
                            self.feature_names[f] + "<=" + \
                            str(round(mma, 3)) + " & "
            if to_vis:
                return [rule[:-3] + " then " + decision, original_number_of_rules, original_number_of_features, len(rules), len(local_feature_names), original_features_cat, local_features_cat, categorical_features_alternatives, feature_rule_limits, original_feature_rule_limits, feature_rule_limits, original_instance]
            del temp_f_maxs, temp_f_mins, f_maxs, f_mins
            if self.mode == 'R':
                return [rule[:-3] + " then " + decision, original_number_of_rules, original_number_of_features, len(rules), len(local_feature_names), original_features_cat, local_features_cat, categorical_features_alternatives, feature_rule_limits, local_error]
            return [rule[:-3] + " then " + decision, original_number_of_rules, original_number_of_features, len(rules), len(local_feature_names), original_features_cat, local_features_cat, categorical_features_alternatives, feature_rule_limits]

        else:
            for ranked_f in self.ranked_features[class_name]:
                f = self.feature_names.index(ranked_f)
                if self.feature_names[f] in local_feature_names:
                    if self.utilizer is not None:
                        mmi = self.utilizer.inverse_transform(
                            np.array([f_mins, f_mins]))[0][f]
                        mma = self.utilizer.inverse_transform(
                            np.array([f_maxs, f_maxs]))[0][f]
                    else:
                        mmi = np.array([f_mins, f_mins])[0][f]
                        mma = np.array([f_maxs, f_maxs])[0][f]  # ena tab mesa
                    feature_rule_limits[self.feature_names[f]] = [mmi, mma]
                    rule = rule + str(round(mmi, 3)) + "<=" + \
                        self.feature_names[f] + "<=" + \
                        str(round(mma, 3)) + " & "
            if to_vis:
                return [rule[:-3] + " then " + decision, original_number_of_rules, original_number_of_features, len(rules), len(local_feature_names), feature_rule_limits, original_feature_rule_limits, feature_rule_limits, original_instance]
        del temp_f_maxs, temp_f_mins, f_maxs, f_mins
        if self.mode == 'R':
            return [rule[:-3] + " then " + decision, original_number_of_rules, original_number_of_features, len(rules), len(local_feature_names), feature_rule_limits, local_error]
        return [rule[:-3] + " then " + decision, original_number_of_rules, original_number_of_features, len(rules), len(local_feature_names), feature_rule_limits]

    def _reduce_rules(self, rules, probabilities, instance_qe, number_of_clusters, ar_algorithm, cl_algorithm, method='123', save_plots=False, instance=None):
        """following_breadcrumbs function finds
        Args:
            instance: The instance we want to find the paths
            reduction: The targets of our train data
            save_plots: The bar and stacked area plots for every feature will be saved
        Return:

        """
        reduced_rules = rules
        reduced_probabilities = probabilities
        if ('1' in method and self.mode == 'C') or 'R1' in method:
            # Step 1: Starting with Association rules or R1: Association Rules and Random Selection
            reduced_rules_t, reduced_probabilities_t = self._reduce_through_association_rules(
                reduced_rules, reduced_probabilities, instance_qe, ar_algorithm, save_plots)
            # In case AR reduce the paths to less than a quorum, we reset all the paths.
            if self.mode == 'R' or(np.array(reduced_probabilities_t).sum()/self.number_of_estimators > 0.5 and len(reduced_rules_t) >= instance_qe):
                reduced_rules = reduced_rules_t
                reduced_probabilities = reduced_probabilities_t
        if '2' in method and self.mode == 'C':
            # Step 2: Reduction through Clustering
            reduced_rules_t, reduced_probabilities_t = self._reduce_through_clustering(
                reduced_rules, reduced_probabilities, instance_qe, cl_algorithm, number_of_clusters)
            if np.array(reduced_probabilities_t).sum()/self.number_of_estimators > 0.5 and len(reduced_rules_t) >= instance_qe:
                reduced_rules = reduced_rules_t
                reduced_probabilities = reduced_probabilities_t
        if '3' in method and self.mode == 'C':
            # Step 3: Random Selection of the quorum
            reduced_rules_t, reduced_probabilities_t = self._reduce_through_random_selection(
                reduced_rules, reduced_probabilities, instance_qe, ar_algorithm, cl_algorithm)
            if np.array(reduced_probabilities_t).sum()/self.number_of_estimators > 0.5 and len(reduced_rules_t) >= instance_qe:
                reduced_rules = reduced_rules_t
                reduced_probabilities = reduced_probabilities_t
        if 'R2' in method or 'R3' in method:
            # R2 and R3: Inner and Outter Distribution Based Reduction
            reduced_rules_t, reduced_probabilities_t = self._reduce_through_distribution(instance,
                                                                                         reduced_rules, reduced_probabilities, instance_qe, method)
            reduced_rules = reduced_rules_t
            reduced_probabilities = reduced_probabilities_t
        items = set()
        for pr in reduced_rules:
            for p in pr:
                items.add(p)
        new_feature_list = list(items)
        if self.mode == 'R':
            local_error = abs(np.mean(probabilities) -
                              np.mean(reduced_probabilities))
            return [reduced_rules, new_feature_list, local_error]
        return [reduced_rules, new_feature_list]

    def _reduce_through_association_rules(self, rules, probabilities, instance_qe, ar_algorithm, save_plots=False):
        reduced_rules = rules
        reduced_probabilities = probabilities
        # If we need more reduction on path level
        if (self.mode == 'R' and instance_qe > 0) or (np.array(reduced_probabilities).sum()/self.number_of_estimators > 0.5 and len(reduced_rules) > instance_qe):

            get_itemsets = []
            items = set()
            for pr in rules:
                itemset = []
                for p in pr:
                    itemset.append(p)
                    items.add(p)
                get_itemsets.append(itemset)
            max_number_of_features = len(items)
            del items

            te = TransactionEncoder()
            te_ary = te.fit(get_itemsets).transform(get_itemsets)
            df = pd.DataFrame(te_ary, columns=te.columns_)
            # Run AR algorithm
            if ar_algorithm == 'apriori':
                temp_fi = apriori(df, min_support=0.1, use_colnames=True)
                if len(temp_fi.values) == 0:
                    return rules, probabilities
            else:
                temp_fi = fpgrowth(df, min_support=0.1, use_colnames=True)
                if len(temp_fi.values) == 0:
                    return rules, probabilities
            frequent_itemsets = association_rules(temp_fi, metric="support",
                                                  min_threshold=0.1).sort_values(by="confidence", ascending=True)

            # Collect features from the association rules extracted
            probability = 0
            k = 1
            antecedents = []
            if save_plots:
                size_of_ar = len(list(list(frequent_itemsets['antecedents'])))
            else:
                antecedents_weights = {}
                antecedents_set = set()
                wcounter = 0
                for antecedent in list(frequent_itemsets['antecedents']):
                    if tuple(antecedent) not in antecedents_set:
                        antecedents_set.add(tuple(antecedent))
                        for antecedent_i in list(antecedent):
                            if antecedent_i not in antecedents:
                                antecedents.append(antecedent_i)
                    for antecedent_i in list(antecedent):
                        wcounter = wcounter + 1
                        if antecedent_i not in antecedents_weights:
                            antecedents_weights[antecedent_i] = 1/wcounter
                        else:
                            antecedents_weights[antecedent_i] = antecedents_weights[antecedent_i] + 1/wcounter
                self.silly_local_importance = antecedents_weights
                size_of_ar = len(antecedents)

            items = set()
            new_feature_list = []
            for pr in reduced_rules:
                for p in pr:
                    items.add(p)
            new_feature_list = list(items)
            reduced_rules = []
            reduced_probabilities = []
            if self.mode == 'R':
                local_error = 2 * abs(instance_qe)
                keep_pids = []
                # print('local error 0:', local_error, abs(instance_qe))
                while local_error > abs(instance_qe) and k <= size_of_ar:
                    feature_set = set()
                    if save_plots:
                        for i in range(0, k):
                            for j in list(list(frequent_itemsets['antecedents'])[i]):
                                feature_set.add(j)
                    else:
                        for i in range(0, k):
                            feature_set.add(antecedents[i])
                    new_feature_list = list(feature_set)
                    redundant_features = [
                        i for i in self.feature_names if i not in new_feature_list]
                    reduced_rules = []
                    pid = 0
                    keep_pids = []
                    reduced_probabilities = []
                    for i in rules:
                        if sum([1 for j in redundant_features if j in i]) == 0:
                            reduced_rules.append(i)
                            reduced_probabilities.append(probabilities[pid])
                            keep_pids.append(pid)
                        else:
                            dis_a = abs(
                                probabilities[pid] - self.min_max_leaf_prediction_per_trees[pid][0][0])
                            dis_b = abs(
                                probabilities[pid] - self.min_max_leaf_prediction_per_trees[pid][1][0])
                            if dis_a < dis_b:
                                reduced_probabilities.append(
                                    self.min_max_leaf_prediction_per_trees[pid][1][0])
                            else:
                                reduced_probabilities.append(
                                    self.min_max_leaf_prediction_per_trees[pid][0][0])
                        pid = pid + 1
                    local_error = abs(np.mean(probabilities) -
                                      np.mean(reduced_probabilities))
                    # print('ar', local_error, len(keep_pids), len(
                    #    reduced_probabilities)-len(keep_pids))
                    k += 1
                if local_error > abs(instance_qe) and k > size_of_ar:
                    keep_pids = []
                    reduced_rules = []
                    reduced_probabilities = []
                    pid = 0
                    for i in rules:
                        reduced_rules.append(i)
                        reduced_probabilities.append(probabilities[pid])
                        keep_pids.append(pid)
                        pid = pid + 1
                temp_pids = keep_pids.copy()
                last_pid = None
                # print('local error 1:', local_error,
                #      abs(instance_qe), len(keep_pids))
                while local_error < abs(instance_qe) and len(temp_pids) > 2:
                    reduced_rules = []
                    pid = 0
                    reduced_probabilities = []
                    last_pid = temp_pids[-1]
                    temp_pids = temp_pids[:-1]
                    for i in rules:
                        if pid in temp_pids:
                            reduced_rules.append(i)
                            reduced_probabilities.append(probabilities[pid])
                        else:
                            dis_a = abs(
                                probabilities[pid] - self.min_max_leaf_prediction_per_trees[pid][0][0])
                            dis_b = abs(
                                probabilities[pid] - self.min_max_leaf_prediction_per_trees[pid][1][0])
                            if dis_a < dis_b:
                                reduced_probabilities.append(
                                    self.min_max_leaf_prediction_per_trees[pid][1][0])
                            else:
                                reduced_probabilities.append(
                                    self.min_max_leaf_prediction_per_trees[pid][0][0])
                        pid = pid + 1
                    local_error = abs(np.mean(probabilities) -
                                      np.mean(reduced_probabilities))
                    # print('random:', local_error)
                if last_pid is not None:
                    temp_pids.append(last_pid)
                    reduced_rules = []
                    pid = 0
                    reduced_probabilities = []
                    for i in rules:
                        if pid in temp_pids:
                            reduced_rules.append(i)
                            reduced_probabilities.append(probabilities[pid])
                        else:
                            dis_a = abs(
                                probabilities[pid] - self.min_max_leaf_prediction_per_trees[pid][0][0])
                            dis_b = abs(
                                probabilities[pid] - self.min_max_leaf_prediction_per_trees[pid][1][0])
                            if dis_a < dis_b:
                                reduced_probabilities.append(
                                    self.min_max_leaf_prediction_per_trees[pid][1][0])
                            else:
                                reduced_probabilities.append(
                                    self.min_max_leaf_prediction_per_trees[pid][0][0])
                        pid = pid + 1
                    local_error = abs(np.mean(probabilities) -
                                      np.mean(reduced_probabilities))
                # print('local error 2:', local_error)

            else:
                while (probability <= 0.5 or len(reduced_rules) < instance_qe) and k < size_of_ar:
                    feature_set = set()
                    if save_plots:
                        for i in range(0, k):
                            for j in list(list(frequent_itemsets['antecedents'])[i]):
                                feature_set.add(j)
                    else:
                        for i in range(0, k):
                            feature_set.add(antecedents[i])
                    new_feature_list = list(feature_set)
                    redundant_features = [
                        i for i in self.feature_names if i not in new_feature_list]
                    reduced_rules = []
                    pid = 0
                    reduced_probabilities = []
                    for i in rules:
                        if sum([1 for j in redundant_features if j in i]) == 0:
                            reduced_rules.append(i)
                            reduced_probabilities.append(probabilities[pid])
                        pid = pid + 1
                    probability = np.array(
                        reduced_probabilities).sum()/self.number_of_estimators
                    k += 1
        return reduced_rules, reduced_probabilities

    def _reduce_through_clustering(self, rules, probabilities, instance_qe, cl_algorithm, number_of_clusters):
        reduced_rules = rules
        reduced_probabilities = probabilities

        # Step 2: Clustering
        # If we need more reduction on path level
        if np.array(reduced_probabilities).sum()/self.number_of_estimators > 0.5 and len(reduced_rules) > instance_qe:
            # We need to build the similarity matrix using our path similarity metric
            items = set()
            for pr in reduced_rules:
                for p in pr:
                    items.add(p)
            new_feature_list = list(items)
            similarity_matrix = []
            for k in range(len(reduced_rules)):
                B = []
                for j in range(len(reduced_rules)):
                    if k == j:
                        B.append(0)
                    else:
                        sim = path_similarity(reduced_rules[k], reduced_rules[j], new_feature_list,
                                              self.min_max_feature_values)
                        if cl_algorithm == 'kmedoids':
                            B.append(1 - sim)
                            # B.append(sim)
                        else:
                            B.append(sim)
                similarity_matrix.append(B)
            similarity_matrix = np.array(similarity_matrix)
            if cl_algorithm == 'kmedoids':
                # Then we run our clustering algorithm
                try:
                    MS, clusters = kMedoids(
                        similarity_matrix, number_of_clusters)
                    clusters_sorted = sorted(
                        clusters, key=lambda k: len(clusters[k]), reverse=True)
                except:
                    # print("Warning! Error occured on kMedoids computation of this instance.")
                    clusters_sorted = None
            elif cl_algorithm == 'OPTICS':
                try:
                    clustering = OPTICS(min_samples=number_of_clusters, metric='precomputed').fit(
                        similarity_matrix)
                    x = clustering.labels_
                    set(x)
                    clusters = {}
                    for i in set(x):
                        clusters[i] = []
                    for i in range(len(x)):
                        clusters[x[i]].append(i)
                    clusters_sorted = sorted(
                        clusters, key=lambda k: len(clusters[k]), reverse=True)
                except:
                    # print("Warning! Error occured on OPTICS computation of this instance.")
                    clusters_sorted = None
            else:  # if cl_algorithm == 'SC':
                try:
                    clustering = SpectralClustering(
                        n_clusters=number_of_clusters, affinity='precomputed', random_state=0).fit(similarity_matrix)
                    x = clustering.labels_
                    set(x)
                    clusters = {}
                    for i in set(x):
                        clusters[i] = []
                    for i in range(len(x)):
                        clusters[x[i]].append(i)
                    clusters_sorted = sorted(
                        clusters, key=lambda k: len(clusters[k]), reverse=True)
                except:
                    # print("Warning! Error occured on SC computation of this instance.")
                    clusters_sorted = None
                    clusters = {}
            # If no error occured then we procceed to the path collection
            if clusters_sorted is not None:
                k = 0
                size = 0
                probability = 0
                reduced_rules = []
                reduced_probabilities = []
                while (probability <= 0.5 or len(reduced_rules) < instance_qe) and k < len(clusters_sorted):
                    for j in clusters[clusters_sorted[k]]:
                        reduced_rules.append(rules[j])
                        reduced_probabilities.append(probabilities[j])
                    k += 1
                    size = len(reduced_rules)
                    probability = np.array(
                        reduced_probabilities).sum()/self.number_of_estimators
        return reduced_rules, reduced_probabilities

    def _reduce_through_random_selection(self, rules, probabilities, instance_qe, ar_algorithm, cl_algorithm):
        reduced_rules = rules
        reduced_probabilities = probabilities

        if np.array(reduced_probabilities).sum()/self.number_of_estimators > 0.5 and len(reduced_rules) >= instance_qe:
            random.seed(2000+len(str(ar_algorithm)) +
                        len(str(cl_algorithm))+len(reduced_rules))
            random.shuffle(reduced_rules)
            reduced_rules_t = reduced_rules
            reduced_probabilities_t = reduced_probabilities
            c_remove = 0
            while np.array(reduced_probabilities_t).sum()/self.number_of_estimators > 0.5 and len(reduced_rules_t) >= instance_qe:
                reduced_rules_t = reduced_rules_t[:-1]
                reduced_probabilities_t = reduced_probabilities_t[:-1]
                c_remove = c_remove + 1
            if c_remove > 1:
                c_remove = c_remove - 1
                reduced_rules = reduced_rules[:-c_remove]
                reduced_probabilities = reduced_probabilities[:-c_remove]
        return reduced_rules, reduced_probabilities

    def _reduce_through_distribution(self, instance, rules, predictions, instance_qe, method):
        reduced_rules = rules
        reduced_predictions = predictions
        real_prediction = np.array(predictions).mean()
        min_errors = abs(instance_qe)
        min_s = 0
        for s in [.1, .2, .5, 1, 2, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100]:
            np.random.seed(int(instance.sum()*(instance.sum()/2)/10))
            normal_dis = np.random.normal(
                real_prediction, np.array(predictions).std()/s, 100)

            c = 0
            l_reduced_rules = []
            l_reduced_predictions = []

            for i in predictions:
                # R2 is inner and R3 is outter
                if (method == 'R2' and not (i < normal_dis.min() or i > normal_dis.max())) or (method == 'R3' and (i < normal_dis.min() or i > normal_dis.max())):
                    l_reduced_rules.append(rules[c])
                    l_reduced_predictions.append(predictions[c])
                else:
                    dis_a = abs(
                        i - self.min_max_leaf_prediction_per_trees[c][0][0])
                    dis_b = abs(
                        i - self.min_max_leaf_prediction_per_trees[c][1][0])
                    if dis_a < dis_b:
                        l_reduced_predictions.append(
                            self.min_max_leaf_prediction_per_trees[c][1][0])
                    else:
                        l_reduced_predictions.append(
                            self.min_max_leaf_prediction_per_trees[c][0][0])
                c = c + 1
            l_error = abs(
                np.array(l_reduced_predictions).mean() - real_prediction)
            # print(s, l_error, len(l_reduced_rules))
            if l_error < abs(instance_qe) and len(l_reduced_rules) < len(reduced_rules):
                # print('new best')
                reduced_rules = l_reduced_rules
                reduced_predictions = l_reduced_predictions
                min_errors = l_error
                min_s = s

        return reduced_rules, reduced_predictions

    def _pre_feature_range_caluclation(self, rules, feature):
        mi = None
        ma = None
        for i in rules:
            if feature in i:
                if len(i[feature]) == 1:
                    if i[feature][0][0] == "<=":
                        if ma is None or ma >= i[feature][0][1]:
                            ma = i[feature][0][1]
                    else:
                        if mi == None or mi <= i[feature][0][1]:
                            mi = i[feature][0][1]
                else:
                    if mi == None or mi <= i[feature][1][1]:
                        mi = i[feature][1][1]
                    if ma == None or ma >= i[feature][0][1]:
                        ma = i[feature][0][1]
        if mi is None:
            mi = self.min_max_feature_values[feature][0]
        if ma is None:
            ma = self.min_max_feature_values[feature][1]
        return [mi, ma]

    def _pre_feature_range_caluclation_bars(self, rules, feature, complexity=4):
        mi = self.min_max_feature_values[feature][0]
        ma = self.min_max_feature_values[feature][1]
        for i in rules:
            if feature in i:
                if len(i[feature]) == 1:
                    if i[feature][0][0] == "<=":
                        if ma < i[feature][0][1]:
                            ma = i[feature][0][1]
                    else:
                        if mi > i[feature][0][1]:
                            mi = i[feature][0][1]
                else:
                    if mi > i[feature][1][1]:
                        mi = i[feature][1][1]
                    if ma < i[feature][0][1]:
                        ma = i[feature][0][1]

        bars = []
        temp_count = 0
        for i in rules:
            if feature in i:
                temp_count += 1
                if len(i[feature]) == 1:
                    if i[feature][0][0] == "<=":
                        bars.append(np.arange(roundup(mi, complexity), roundup(i[feature][0][1], complexity),
                                              (10 ** (-complexity))))
                    else:
                        bars.append(np.arange(roundup(i[feature][0][1], complexity), roundup(ma, complexity),
                                              (10 ** (-complexity))))
                else:
                    mm = [roundup(i[feature][0][1], complexity),
                          roundup(i[feature][1][1], complexity)]
                    bars.append(
                        np.arange(min(mm), max(mm), (10 ** (-complexity))))
        if temp_count == 0:
            return False, False
        bars_len = [len(bar) for bar in bars]
        return bars, bars_len

    def _aggregated_feature_range(self, bars, feature, save_plots=False, complexity=4):
        """_aggregated_feature_range function returns the min and max value from the intersection of all paths
        Args:
            feature: the feature which range we want to find
            save_plots: if yes then it will save the bar and stacked area plots of each feature
            complexity: determines how many digits we will use to descritize
        Return:
            min max of the intersect area of all paths for a feature
        """
        mi = self.min_max_feature_values[feature][0]
        ma = self.min_max_feature_values[feature][1]

        if save_plots:
            plt.figure(figsize=(16, 10))
            plt.title(feature)
            plt.ylabel('No of Rules')
            plt.xlabel('Value')
            for i in range(len(bars)):
                plt.plot(bars[i], len(bars[i]) * [i + 1], linewidth=8)
            plt.savefig(feature + "BarsPlot.png")

        temp_bars = []
        for i in bars:
            bar = set()
            for j in i:
                bar.add(
                    int((roundup(j, complexity) - roundup(mi, complexity)) * (10 ** complexity)))
            temp_bars.append(bar)
        bars = temp_bars
        del temp_bars

        st = {}
        for i in bars:
            for j in i:
                if not int(j) in st:
                    st[int(j)] = 1
                else:
                    st[int(j)] += 1
        del bars

        x = []
        y = []
        max_v = -1
        for key, value in st.items():
            if max_v < value:
                max_v = value
            x.append((key + int(roundup(mi, complexity) *
                                (10 ** complexity))) / (10 ** complexity))
            y.append(value)
        x, y = zip(*sorted(zip(x, y)))
        x_2 = []
        x_3 = []
        y_2 = []
        for key, value in st.items():
            x_2.append((key + int(roundup(mi, complexity) *
                                  (10 ** complexity))) / (10 ** complexity))
            if max_v == value:
                x_3.append((key + int(roundup(mi, complexity) *
                                      (10 ** complexity))) / (10 ** complexity))
                y_2.append(value)
            else:
                y_2.append(0)
        del st

        if save_plots:
            x_2, y_2 = zip(*sorted(zip(x_2, y_2)))
            plt.figure(figsize=(16, 10))
            plt.title(feature)
            plt.ylabel('No of Rules')
            plt.xlabel('Value')
            plt.stackplot(x, y)
            plt.stackplot(x, y_2, colors='c')
            plt.savefig(feature + "StackedAreaPlot.png")
        del x, y, x_2, y_2
        return [min(x_3), max(x_3)]

    def _show_distribution_plots(self, index):
        fig = self.distribution_plots[index]
        dummy = plt.figure()
        new_manager = dummy.canvas.manager
        new_manager.canvas.figure = fig
        fig.set_canvas(new_manager.canvas)

    def _plot_visuals(self, f_sldr, t_ori):
        inst_stats = self.inst_stats
        # CHECK IF CAT BCAUSE IS DIF
        index = f_sldr - 1
        if self.categorical_features is not None:
            avail_features = self.numerical_feature_names + self.categorical_features
            print('Feature:', avail_features[index])
            if avail_features[index] in self.numerical_feature_names:
                if avail_features[index] in inst_stats[-2].keys():
                    index = self.feature_names.index(avail_features[index])
                    buf = io.BytesIO()
                    pickle.dump(self.distribution_plots[index], buf)
                    buf.seek(0)
                    fig = pickle.load(buf)

                    dummy = plt.figure(num=None, figsize=(
                        10, 7), dpi=100, facecolor='w', edgecolor='k')
                    new_manager = dummy.canvas.manager
                    new_manager.canvas.figure = fig
                    fig.set_canvas(new_manager.canvas)

                    if t_ori:
                        plt.axvline(
                            x=inst_stats[-3][self.feature_names[index]][0], color='b')
                        plt.axvline(
                            x=inst_stats[-3][self.feature_names[index]][1], color='b')
                    plt.axvline(
                        x=inst_stats[-2][self.feature_names[index]][0], color='g')
                    plt.axvline(
                        x=inst_stats[-2][self.feature_names[index]][1], color='g')
                    plt.axvline(x=inst_stats[-1][index], color='r')
                    plt.show()
                else:
                    print("This feature does not appear in the explanation!")

            else:
                if avail_features[index] not in self.inst_stats[0]:
                    for k, v in self.inst_stats[-5].items():
                        if avail_features[index] in k:
                            print(
                                " If you change this feature's value with one of the following the prediction might change:")
                            for c in v:
                                print('    ', c)
        else:
            print('My life sucks')
            if self.feature_names[index] in inst_stats[-2].keys():
                buf = io.BytesIO()
                pickle.dump(self.distribution_plots[index], buf)
                buf.seek(0)
                fig = pickle.load(buf)

                dummy = plt.figure(num=None, figsize=(
                    10, 7), dpi=100, facecolor='w', edgecolor='k')
                new_manager = dummy.canvas.manager
                new_manager.canvas.figure = fig
                fig.set_canvas(new_manager.canvas)

                if t_ori:
                    plt.axvline(
                        x=inst_stats[-3][self.feature_names[index]][0], color='b')
                    plt.axvline(
                        x=inst_stats[-3][self.feature_names[index]][1], color='b')
                plt.axvline(
                    x=inst_stats[-2][self.feature_names[index]][0], color='g')
                plt.axvline(
                    x=inst_stats[-2][self.feature_names[index]][1], color='g')
                plt.axvline(x=inst_stats[-1][index], color='r')
                plt.show()
            else:
                print("This feature does not appear in the explanation!")

    def visualize(self, instance_stats):
        '''Setting up the interactive visualization tool'''

        # UI elements
        feature_slider = IntSlider(min=1, max=len(
            self.feature_names), default_value=1, description="Feature: ", continuous_update=False)
        if self.categorical_features is not None:
            avail_features = self.numerical_feature_names + self.categorical_features
            feature_slider = IntSlider(min=1, max=len(
                avail_features), default_value=1, description="Feature: ", continuous_update=False)
        original = ToggleButtons(
            options=['Yes', 'No'], description="Original ranges:")
        print('Prediction rule:', instance_stats[0])
        ui = HBox([feature_slider, original])
        self.inst_stats = instance_stats
        # Starting the interactive tool
        inter = interactive_output(
            self._plot_visuals, {'f_sldr': feature_slider, 't_ori': original})
        display(ui, inter)

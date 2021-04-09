from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

class GlobalSurrogateTree:
    def __init__(self, x, y, feature_names, task):
        self.feature_names = feature_names
        if task=='classification':
            dtree = DecisionTreeClassifier(random_state = 10)
            parameters = [{
                'criterion': ['gini','entropy'],
                'splitter': ['best','random'],
                'max_depth': [1, 2, 5, 10, None],
                'max_features': ['sqrt', 'log2', 0.75, None],#['sqrt', 'log2', 0.75, None], #'sqrt', 'log2', 0.75, None
                'min_samples_leaf' : [1, 2, 5, 10],#[1, 2, 5, 10, 0.10], #1, 2, 5, 10, 0.10
            }]
            clf = GridSearchCV(estimator=dtree, param_grid=parameters, cv=10, n_jobs=-1, verbose=0, scoring='f1_weighted')
        else:
            dtree = DecisionTreeRegressor(random_state = 10)
            parameters = [{
                'criterion': ['mse', 'friedman_mse', 'mae', 'poisson'],
                'splitter': ['best','random'],
                'max_depth': [1, 2, 5, 10, None],
                'max_features': ['sqrt', 'log2', 0.75, None],#['sqrt', 'log2', 0.75, None], #'sqrt', 'log2', 0.75, None
                'min_samples_leaf' : [1, 2, 5, 10],#[1, 2, 5, 10, 0.10], #1, 2, 5, 10, 0.10
            }]
            clf = GridSearchCV(estimator=dtree, param_grid=parameters, cv=10, n_jobs=-1, verbose=0, scoring='neg_mean_absolute_error')
        clf.fit(x, y)
        self.accuracy = clf.best_score_
        self.model = clf.best_estimator_
        
    def rule(self,instance):
        path = self.model.decision_path([instance])
        leq = {}  # leq: less equal ex: x <= 1
        b = {}  # b: bigger ex: x > 0.6
        local_range = {}
        for node in path.indices:
            feature_id = self.model.tree_.feature[node]
            feature = self.feature_names[feature_id]
            threshold = self.model.tree_.threshold[node]
            if threshold != -2.0:
                if instance[feature_id] <= threshold:
                    leq.setdefault(feature, []).append(threshold)
                else:
                    b.setdefault(feature, []).append(threshold)
        for k in leq:
            local_range.setdefault(k, []).append(['<=', min(leq[k])])  # !!
        for k in b:
            local_range.setdefault(k, []).append(['>', max(b[k])])  # !!
        return local_range, self.model.predict([instance])[0]
    
class LocalSurrogateTree:
    def __init__(self, x, y, feature_names, task, neighbours=None):
        self.x = x
        self.y = y
        self.neighbours = neighbours
        if neighbours is None:
            self.neighbour = int(len(x)/10)
        neighbours_generator = KNeighborsClassifier(n_neighbors=self.neighbours, weights="distance", metric="minkowski", p=2)
        neighbours_generator.fit(self.x, self.y)
        self.neighbours_generator = neighbours_generator
        self.feature_names = feature_names
        if task=='classification':
            dtree = DecisionTreeClassifier(random_state = 10)
            parameters = [{
                'criterion': ['gini','entropy'],
                'splitter': ['best','random'],
                'max_depth': [1, 2, 5, 10, None],
                'max_features': ['sqrt', 'log2', 0.75, None],#['sqrt', 'log2', 0.75, None], #'sqrt', 'log2', 0.75, None
                'min_samples_leaf' : [1, 2, 5, 10],#[1, 2, 5, 10, 0.10], #1, 2, 5, 10, 0.10
            }]
            self.clf = GridSearchCV(estimator=dtree, param_grid=parameters, cv=10, n_jobs=-1, verbose=0, scoring='f1_weighted')
        else:
            dtree = DecisionTreeRegressor(random_state = 10)
            parameters = [{
                'criterion': ['mse', 'friedman_mse', 'mae', 'poisson'],
                'splitter': ['best','random'],
                'max_depth': [1, 2, 5, 10, None],
                'max_features': ['sqrt', 'log2', 0.75, None],#['sqrt', 'log2', 0.75, None], #'sqrt', 'log2', 0.75, None
                'min_samples_leaf' : [1, 2, 5, 10],#[1, 2, 5, 10, 0.10], #1, 2, 5, 10, 0.10
            }]
            self.clf = GridSearchCV(estimator=dtree, param_grid=parameters, cv=10, n_jobs=-1, verbose=0, scoring='neg_mean_absolute_error')
    def _generate_neighbours(self,instance):
        x = [instance]
        ys = self.neighbours_generator.kneighbors(x, n_neighbors=self.neighbours, return_distance=False)
        new_x_train = []
        new_y_train = []
        for i in ys[0]:
            new_x_train.append(self.x[i])
            new_y_train.append(self.y[i])
        return new_x_train, new_y_train
    
    def rule(self,instance):
        local_x, local_y = self._generate_neighbours(instance)
        self.clf.fit(local_x, local_y)
        model = self.clf.best_estimator_
        
        path = model.decision_path([instance])
        leq = {}  # leq: less equal ex: x <= 1
        b = {}  # b: bigger ex: x > 0.6
        local_range = {}
        for node in path.indices:
            feature_id = model.tree_.feature[node]
            feature = self.feature_names[feature_id]
            threshold = model.tree_.threshold[node]
            if threshold != -2.0:
                if instance[feature_id] <= threshold:
                    leq.setdefault(feature, []).append(threshold)
                else:
                    b.setdefault(feature, []).append(threshold)
        for k in leq:
            local_range.setdefault(k, []).append(['<=', min(leq[k])])  # !!
        for k in b:
            local_range.setdefault(k, []).append(['>', max(b[k])])  # !!
        return local_range, model.predict([instance])[0]
import pandas as pd
import numpy as np
from scipy import stats
import urllib
from sklearn.preprocessing import OneHotEncoder


class Dataset:
    def __init__(self, x=None, y=None, feature_names=None, class_names=None, categorical_features=None):
        self.x = x
        self.y = y
        self.feature_names = feature_names
        self.categorical_features = categorical_features
        self.class_names = class_names
        
    def load_boston(self):
        from sklearn.datasets import load_boston
        data = load_boston()
        self.x = data['data']
        self.y = data['target']
        self.feature_names = data['feature_names']
        self.class_names=["House Price"]
        return self.x, self.y, self.feature_names, self.class_names

    def load_wine_quality(self):
        df1 = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', delimiter=';')
        df2 = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv', delimiter=';')
        self.feature_names = list(df1.columns.values)[:-1]
        self.class_names = list(df1.columns.values)[-1:]
        x1 = df1.values[:,:-1]
        y1 = df1.values[:,-1:]
        x2 = df2.values[:,:-1]
        y2 = df2.values[:,-1:]
        x = []
        y = []
        for i in range(len(x1)):
            x.append(x1[i])
            y.append(y1[i])
        for i in range(len(x2)):
            x.append(x2[i])
            y.append(y2[i])
        self.x = np.array(x)
        self.y = np.array([i[0] for i in y])
        return self.x, self.y, self.feature_names, self.class_names
            
    def load_segment(self):
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/segment/segment.dat'
        self.feature_names = ['region-centroid-col', 'region-centroid-row', 'region-pixel-count', 'short-line-density-5',
                              'short-line-density-2', 'vedge-mean', 'vegde-sd', 'hedge-mean', 'hedge-sd', 'intensity-mean',
                              'rawred-mean', 'rawblue-mean', 'rawgreen-mean', 'exred-mean', 'exblue-mean', 'exgreen-mean',
                              'value-mean', 'saturatoin-mean', 'hue-mean']

        self.class_names = ['brickface', 'sky', 'foliage',
                            'cement', 'window', 'path', 'grass']
        segment_data = pd.read_csv(
            url, names=self.feature_names+['class'], delimiter=' ')
        self.x = segment_data.values[:, :-1]
        y = segment_data.values[:, -1:]
        self.y = [int(i - 1) for i in y]
        return self.x, self.y, self.feature_names, self.class_names

    def load_abalone(self, type='classification', ys='all'):
        # used this notebook as reference: https://www.kaggle.com/princeashburton/abalone-analysis-supervised-learning/notebook
        self.feature_names = ['Sex', 'Length', 'Diam', 'Height',
                              'Whole', 'Shucked', 'Viscera', 'Shell', 'Rings']
        self.categorical_features = ['Sex']
        abalone = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data',
                              names=self.feature_names)
        # We can use the Scipy stats module to convert the observations into Zscores. Zscores give us a distribution
        # that illustrate how many standard deviations away from the mean the data is.
        z = np.abs(stats.zscore(abalone.select_dtypes(include=[np.number])))
        abalone_o = abalone[(z < 3).all(axis=1)]
        self.x = abalone_o.values[:, :-1]
        for i in self.x:
            if i[0] == 'M':
                i[0] = 0
            else:
                i[0] = 1
        self.y = abalone_o.Rings.values
        if type == 'regression':
            self.class_names = ['Age (+1.5)']
        else:
            # Classiciation
            if ys == 'all':
                class_names = []
                for i in set(self.y):
                    class_names.append('Age_'+str(i))
                self.class_names = class_names
            else:  # reduced
                y_classification = []
                for i in self.y:
                    if i <= 5:
                        y_classification.append(0)  # '(,6.5]'
                    elif i <= 10:
                        y_classification.append(1)  # '(6.5,11.5]'
                    elif i <= 15:
                        y_classification.append(2)  # '(11.5-16.5]'
                    else:
                        y_classification.append(3)  # '(16.5,)'
                self.class_names = ['(,6.5]', '(6.5,11.5]',
                                    '(11.5-16.5]', '(16.5,)']
                self.y = y_classification
        return self.x, self.y, self.feature_names[:-1], self.class_names

    def load_glass(self):
        """
            This method returns the Glass dataset: https://archive.ics.uci.edu/ml/machine-learning-databases/glass
        """

        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
        self.feature_names = ['ID', 'refractive index', 'sodium', 'magnesium', 'aluminum',
                              'silcon', 'potassium', 'calcium', 'barium', 'iron',
                              'class']
        self.class_names = ['building_windows_float_processed', 'building_windows_non_float_processed'                            # ,'vehicle_windows_float_processed' there are no such records...
                            , 'vehicle_windows_non_float_processed (none in this database)', 'containers', 'tableware', 'headlamps']
        glass_data = pd.read_csv(url, names=self.feature_names)
        glass_data = glass_data.dropna()
        self.x = glass_data.values[:, 1:-1]
        y = glass_data.values[:, -1:]
        self.y = y.reshape((1, len(y)))[0]
        self.y = self.y - 1
        self.y = [z - 1 if z >= 4 else z for z in self.y]
        return self.x, self.y, self.feature_names[1:-1], self.class_names

    def load_banknote(self):
        """
            This method returns the Banknote dataset: https://github.com/Kuntal-G/Machine-Learning/blob/master/R-machine-learning/data/banknote-authentication.csv
        """
        banknote_datadset = pd.read_csv(
            'https://raw.githubusercontent.com/Kuntal-G/Machine-Learning/master/R-machine-learning/data/banknote-authentication.csv')
        self.x = banknote_datadset.iloc[:, 0:4].values
        self.y = banknote_datadset.iloc[:, 4].values
        self.feature_names = ['variance', 'skew', 'curtosis', 'entropy']
        # 0: no, 1: yes #or ['not authenticated banknote','authenticated banknote']
        self.class_names = ['fake banknote', 'real banknote']
        return self.x, self.y, self.feature_names, self.class_names

    def load_heart(self):
        """
            This method returns the Heart Statlog dataset: https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart
        """

        url = "http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat"
        raw_data = urllib.request.urlopen(url)
        credit = np.genfromtxt(raw_data)
        self.x, self.y = credit[:, :-1], credit[:, -1].squeeze()
        self.y = [int(i-1) for i in self.y]
        self.feature_names = ['age', 'sex', 'chest pain', 'resting blood pressure', 'serum cholestoral',
                              'fasting blood sugar', 'resting electrocardiographic results',
                              'maximum heart rate achieved', 'exercise induced angina', 'oldpeak',
                              'the slope of the peak exercise', 'number of major vessels', 'reversable defect']
        self.class_names = ['absence', 'presence']
        return self.x, self.y, self.feature_names, self.class_names

    def load_adult(self):
        """
            This method returns the Adult dataset: 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
        """
        feature_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                         'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                         'hours-per-week', 'native-country', 'salary']
        self.class_names = ['<=50K', '>50K']  # 0: <=50K and 1: >50K
        data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
                           names=feature_names, delimiter=', ')
        data_test = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test',
                                names=feature_names, delimiter=', ')
        data_test = data_test.drop(data_test.index[[0]])
        data = data[(data != '?').all(axis=1)]
        data_test = data_test[(data_test != '?').all(axis=1)]
        data_test['salary'] = data_test['salary'].map(
            {'<=50K.': '<=50K', '>50K.': '>50K'})
        frames = [data, data_test]
        data = pd.concat(frames)
        hs_grad = ['HS-grad', '11th', '10th', '9th', '12th']
        elementary = ['1st-4th', '5th-6th', '7th-8th']
        # replace elements in list.
        for i in hs_grad:
            data['education'].replace(i, 'HS-grad', regex=True, inplace=True)
        for e in elementary:
            data['education'].replace(
                e, 'Elementary-school', regex=True, inplace=True)
        married = ['Married-spouse-absent',
                   'Married-civ-spouse', 'Married-AF-spouse']
        separated = ['Separated', 'Divorced']
        # replace elements in list.
        for m in married:
            data['marital-status'].replace(m,
                                           'Married', regex=True, inplace=True)
        for s in separated:
            data['marital-status'].replace(s,
                                           'Separated', regex=True, inplace=True)
        self_employed = ['Self-emp-not-inc', 'Self-emp-inc']
        govt_employees = ['Local-gov', 'State-gov', 'Federal-gov']
        for se in self_employed:
            data['workclass'].replace(
                se, 'Self_employed', regex=True, inplace=True)
        for ge in govt_employees:
            data['workclass'].replace(
                ge, 'Govt_employees', regex=True, inplace=True)
        index_age = data[data['age'] == 90].index
        data.drop(labels=index_age, axis=0, inplace=True)
        categorical_data = data[['workclass', 'education', 'marital-status',
                                 'occupation', 'race', 'sex', 'native-country']].values
        numerical_data = data[['age', 'fnlwgt', 'capital-gain',
                               'capital-loss', 'hours-per-week']].values
        y = [0 if i == '<=50K' else 1 for i in data['salary'].values]
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(categorical_data)
        categorical_features = ['workclass', 'education',
                                'marital-status', 'occupation', 'race', 'sex', 'native-country']
        feature_names = []
        for i in range(len(categorical_features)):
            for j in enc.categories_[i]:
                feature_names.append(categorical_features[i]+"_"+j)
        for i in ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']:
            feature_names.append(i)
        self.feature_names = feature_names
        self.categorical_features = categorical_features
        self.y = y
        self.x = np.hstack((enc.transform(categorical_data).A, numerical_data))
        for i in self.x:
            i[80] = float(i[80])
        return self.x, self.y, self.feature_names, self.class_names, self.categorical_features

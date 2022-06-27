import pandas as pd
import numpy as np
from scipy import stats
import urllib
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

class Dataset:
    def __init__(self, x=None, y=None, feature_names=None, class_names=None, categorical_features=None):
        self.x = x
        self.y = y
        self.feature_names = feature_names
        self.categorical_features = categorical_features
        self.class_names = class_names

    def load_flags(self):
        df = pd.read_csv('../datasets/flags.csv')
        self.x = df.values[:,:19]
        self.y = df.values[:,19:]
        self.feature_names = [i.strip(' ') for i in df.columns[:19]]
        self.label_names = df.columns[19:]
        return self.x, self.y, self.feature_names, self.label_names

    def load_foodtrucks(self):
        df = pd.read_csv('../datasets/foodtruck.csv')
        y = df.values[:,21:]
        self.y = np.array([list(i) for i in y])

        num_features = ['frequency numeric', 'expenses numeric',
       'taste numeric', 'hygiene numeric', 'menu numeric',
       'presentation numeric', 'attendance numeric', 'ingredients numeric',
       'place.to.sit numeric', 'takeaway numeric', 'variation numeric',
       'stop.strucks numeric', 'schedule numeric',
       'age.group numeric', 'scholarity numeric', 'average.income numeric',
       'has.work numeric']
        num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean'))])

        one_hot_categorical_features = ['motivation', 'time']
        one_hot_transformer = Pipeline(steps=[('one_hot', OneHotEncoder())])

        ordinal_categorical_features = ['gender', 'marital.status']
        ordinal_transformer = Pipeline(steps=[('ordinal', OrdinalEncoder())])

        preprocess = ColumnTransformer(transformers = [('num', num_transformer,num_features),
                                                       ('one-hot',one_hot_transformer,one_hot_categorical_features),
                                                       ('ordinal',ordinal_transformer,ordinal_categorical_features)])
        preprocess.fit(df)
        x = preprocess.transform(df)
        x[:,-2:] = x[:,-2:]+1
        self.x = x
        self.feature_names = num_features + list(preprocess.named_transformers_['one-hot'].named_steps['one_hot'].get_feature_names()) + ordinal_categorical_features
        self.label_names = df.columns[21:]
        return self.x, self.y, self.feature_names, self.label_names


    def load_water_quality(self):
        df = pd.read_csv('../datasets/water-quality-nom.csv')
        self.x = df.values[:,:16]
        self.y = df.values[:,16:]
        self.feature_names = list(df.columns[:16])
        self.label_names = list(df.columns[16:])
        return self.x, self.y, self.feature_names, self.label_names

    def load_ai4i(self):
        df = pd.read_csv('../datasets/ai4i2020.csv')
        df = df[df['Machine failure'] >0]
        df['Type'] = df['Type'].map(
                    {'L': 1, 'M':2, 'H': 3})
        self.x = df.values[:,2:8]
        self.y= df.values[:,9:-1]
        self.y= [list(i) for i in self.y]
        self.feature_names = list(df.columns[2:8])
        self.label_names = list(df.columns[9:-1])
        return self.x, self.y, self.feature_names, self.label_names

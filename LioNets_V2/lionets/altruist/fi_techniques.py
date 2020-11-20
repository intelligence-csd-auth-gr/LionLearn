import numpy as np
import os
import lime.lime_tabular as lt
from sklearn.inspection import permutation_importance
from eli5 import explain_prediction
from eli5.formatters.as_dataframe import format_as_dataframe
import shap

class FeatureImportance:
    """
    FeatureImportance:  A class containing multiple feature importance interpretation technique for experimentation reasons.
    ...
    
    Methods
    -------
    fi_lime(instance, prediction, model)
        A method that given an instance and a model returns the LIME interpretation for its prediction.
    fi_shap(instance, prediction, model)
        A method that given an instance and a model returns the SHAP interpretation for its prediction.
    fi_eli(instance, prediction, model)
        A method that given an instance and a model returns the Eli5 interpretation for its prediction.
    fi_perm_imp(instance, prediction, model)
        A method that given an instance and a model returns the Permutation Importance interpretation for its prediction.
    fi_coef_lr(instance, prediction, model)
        A method that given an instance and a model returns the Coefficients of Logistic regression interpretation for its prediction.
    fi_rf(instance, prediction, model)
        A method that given an instance and a model returns the Random Forests Pseudo-interpretation for its prediction.
    """
    
    def __init__ (self, training_data, training_targets, feature_names, class_names):
        """
        Parameters
        ----------
            training_data: numpy array
                The data that the machine learning have been trained on
            training_targets: numpy array
                The data that the machine learning have been trained on
            feature_names: list
                The names of the features
            class_names: list
                The names of the classes
        """
        self.training_data = training_data
        self.training_targets = training_targets
        self.training_summary = shap.kmeans(training_data, 10)

        self.feature_names = feature_names
        self.number_of_features = len(feature_names)
        self.class_names = class_names
        
        self.explainer = lt.LimeTabularExplainer(training_data=self.training_data,
            feature_names=self.feature_names, class_names=self.class_names,
            discretize_continuous=True)
        
    def fi_lime(self, instance, prediction, model):
        """
        Parameters
        ----------
            instance: array
                The instance which prediction's interpretation(s) Altruist will investigate.
            prediction: None or any
                it is not used in this function, but kept for consistency
            model:
                The machine learning model made the prediction
                
        Returns
        -------
        list
            The feature importances provided by LIME
        """
        b = self.explainer.explain_instance(instance, model.predict_proba,
            num_features=self.number_of_features, top_labels=1).local_exp
        b = b[list(b.keys())[0]]
        b.sort()
        return [i[1] for i in list(b)]
        
    def fi_shap(self, instance, prediction, model):
        """
        Parameters
        ----------
            instance: array
                The instance which prediction's interpretation(s) Altruist will investigate.
            prediction: None or any
                it is not used in this function, but kept for consistency
            model:
                The machine learning model made the prediction
                
        Returns
        -------
        list
            The feature importances provided by SHAP
        """
        explainer = shap.KernelExplainer(model.predict,self.training_summary)
        return explainer.shap_values(np.array(instance))
        
    def fi_eli(self, instance, prediction, model):
        """
        Parameters
        ----------
            instance: array
                The instance which prediction's interpretation(s) Altruist will investigate.
            prediction: None or any
                it is not used in this function, but kept for consistency
            model:
                The machine learning model made the prediction
                
        Returns
        -------
        list
            The feature importances provided by Eli5
        """
        fn = [i for i in range(len(instance))]
        temp = format_as_dataframe(explain_prediction(model, instance,top=None))
        temp.drop(['target', 'value'],axis=1,inplace=True)
        temp = temp[temp.feature != '<BIAS>']
        def remove_x(x):
            return int(x.replace('x',''))
        temp['feature'] = temp['feature'].apply(remove_x)
        zero = [j for j in fn if j not in temp['feature'].values]
        for z in zero:
            temp = temp.append({'feature':z,'weight':0}, ignore_index=True)
        temp = temp.sort_values(by=['feature'])
        return temp.values[:,1]
        
    def fi_perm_imp(self, instance, prediction, model):
        """
        Parameters
        ----------
            instance: array
                The instance which prediction's interpretation(s) Altruist will investigate.
            prediction: None or any
                it is not used in this function, but kept for consistency
            model:
                The machine learning model made the prediction
                
        Returns
        -------
        list
            The feature importances provided by Permutation Importance
        """
        result = permutation_importance(model, self.training_data, self.training_targets, n_repeats=10, random_state=0)
        return result.importances_mean

    def fi_coef_lr(self, instance, prediction, model):
        """
        Parameters
        ----------
            instance: array
                The instance which prediction's interpretation(s) Altruist will investigate.
            prediction: None or any
                it is not used in this function, but kept for consistency
            model:
                The machine learning model made the prediction
                
        Returns
        -------
        list
            The feature importances provided by the Logistic Regression's coefficients
        """
        return model.coef_[0]*(-1)
        
    def fi_rf(self, instance, prediction, model):
        """
        Parameters
        ----------
            instance: array
                The instance which prediction's interpretation(s) Altruist will investigate.
            prediction: None or any
                it is not used in this function, but kept for consistency
            model:
                The machine learning model made the prediction
                
        Returns
        -------
        list
            The feature importances provided by the pseudo-interpretation of Random Forests
        """
        return model.feature_importances_*(-1)*instance

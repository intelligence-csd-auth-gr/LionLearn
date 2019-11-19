import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.linear_model import RidgeCV
from sklearn.metrics.pairwise import cosine_similarity

class LioNexplainer:
    """Class for interpreting an instance"""

    def __init__(self, explanator=RidgeCV(alphas=[0,1e-3, 1e-2, 1e-1, 1],fit_intercept=[True,False],cv=10), instance=None, train_data=None, target_data=None, feature_names=None):
        """Init function
        Args:
            explanator: The transparent model that is going to be used for the explanation. Default is Ridge Regression Algorithm.
            If you add another model make sure it provides coef_ attribute information
            instance: The instance to explain
            train_data: The neighbourhood of the above instance
            target_data: The predictions of the neural network for this neighbours
            feature_names: The selected features.
        """
        self.explanator = explanator
        self.instance = instance
        self.train_data = train_data
        self.target_data = target_data
        self.feature_names = feature_names
        self.accuracy_r2 = 0
        self.accuracy_mse = 0
        self.fidelity = 0

    def fit_explanator(self):
        """fit_explanator function trains the transparent regression model with the neighbourhood data
        """
        self.explanator.fit(self.train_data, self.target_data)
        y_pred = self.explanator.predict(self.train_data)
        self.accuracy_r2 = r2_score(self.target_data, y_pred)
        self.accuracy_mse = mean_squared_error(self.target_data, y_pred)
        target_data_binary = [0 if a<0.5 else 1 for a in y_pred]
        predicted_data_binary = [0 if a<0.5 else 1 for a in self.target_data]
        self.fidelity = accuracy_score(target_data_binary,predicted_data_binary) #This is wrong. Because we have regression here!

    #In progress!!!
    def print_fidelity(self):
        print("The fidelity of the LioNet in terms of Accuracy Score is:", self.fidelity)
        print("The fidelity of the LioNet in terms of R^2 Score is:", self.accuracy_r2)
        print("The fidelity of the LioNet in terms of Mean Square Error is:", self.accuracy_mse)

    def show_explanation(self):
        """show_explanation function extracts the weights for the features from the transparent trained model
        and it creates a plot explaining the weights for a specific instance.
        """
        weights = self.explanator.coef_
        model_weights = pd.DataFrame({"Instance's Features": list(self.feature_names), "Features' Weights": list(weights[0] * self.instance.A[0])})
        model_weights = model_weights.sort_values(by="Features' Weights", ascending=False)
        model_weights = model_weights[(model_weights["Features' Weights"] != 0)]
        plt.figure(num=None, figsize=(6, 6), dpi=200, facecolor='w', edgecolor='k')
        sns.barplot(x="Features' Weights", y="Instance's Features", data=model_weights)
        plt.xticks(rotation=90)
        plt.show()
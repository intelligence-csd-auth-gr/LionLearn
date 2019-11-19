from LioNexplainers import LioNexplainer
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity


class LioNet:
    """Class for interpreting an instance"""
    def __init__(self, model=None, autoencoder=None, decoder=None, encoder=None, feature_names=None):
        """Init function
        Args:
            model: The trained predictor model
            autoencoder: The trained autoencoder
            decoder: The second half of the autoencoder
            encoder: The first half of the autoencoder
            feature_names: The selected features. The above networks have been trained with these.
        """
        self.model = model
        self.autoencoder = autoencoder
        self.decoder = decoder
        self.encoder = encoder
        self.feature_names = feature_names
        self.instance = None
        self.generated_neighbourhood = []
        self.final_neighbourhood = []
        self.accuracy = 0
        self.neighbourhood_targets = []

    def explain_instance(self, new_instance, normal_distribution=True):
        """Generates the explanation for an instance
        Args:
            new_instance: The instance to explain
            normal_distribution: Sets the distribution of the neighbourhood to be normal (In progress!)
        """
        self.instance = new_instance
        encoded_instance = self.encoder.predict(list(self.instance))
        neighbourhood = self.neighbourhood_generation(encoded_instance)
        import numpy as np
        self.final_neighbourhood = self.decoder.predict(np.array(neighbourhood))
        print("The predictor classified:",self.model.predict(self.instance)[0])
        self.neighbourhood_targets = self.model.predict(self.final_neighbourhood)
        if normal_distribution:
            self.neighbourhood_to_normal_distribution()
        explainer = LioNexplainer(Ridge(), self.instance, self.final_neighbourhood, self.neighbourhood_targets, self.feature_names)
        explainer.fit_explanator()
        explainer.print_fidelity()
        explainer.show_explanation()
        return True

    def neighbourhood_generation(self, encoded_instance):
        """Generates the neighbourhood of an instance
        Args:
            encoded_instance: The instance to generate neighbours
        Return:
            local_neighbourhood: The generated neighbours
        """
        instance = []
        for i in range(0, len(encoded_instance[0])):
            instance.append(encoded_instance[0][i])
        instance_length = len(instance)
        local_neighbourhood = []
        for i in range(0, instance_length): #Multiplying one feature value at a time with
            for m in [0.25,0.5,0,1,2]: # 1/4, 1/2, 0, 1, 2
                gen = instance.copy()
                gen[i] = gen[i] * m
                local_neighbourhood.append(list(gen))
                del gen
        for i in range(0,5):
            local_neighbourhood.append(instance)
        return local_neighbourhood + local_neighbourhood #We do this in order to have a bigger dataset. But there is no difference after all.

    #In Progress
    def neighbourhood_to_normal_distribution(self):
        """Transforms the distribution of the neighbourhood to normal
        """
        old_neighbourhood = self.final_neighbourhood
        old_targets = self.neighbourhood_targets.copy()
        #...

    def print_neighbourhood_labels_distribution(self):
        """Presenting in a plot the distribution of the neighbourhood data
        """
        plt.hist(self.neighbourhood_targets, color='blue', edgecolor='black', bins=int(180 / 5))
        sns.distplot(self.neighbourhood_targets, hist=True, kde=False, bins=int(180 / 5), color='blue', hist_kws={'edgecolor': 'black'})

        plt.title('Histogram of neighbourhood probabilities')
        plt.ylabel('Neighbours')
        plt.xlabel('Prediction Probabilities')
        plt.show()

    def get_neighbourhood_instance_neighbourhood(self, instance):
        """Returns
        """
        self.instance = instance
        encoded_instance = self.encoder.predict(self.instance)
        neighbourhood = self.neighbourhood_generation(encoded_instance)
        import numpy as np
        self.final_neighbourhood = self.decoder.predict(np.array(neighbourhood))
        print("The predictor classified:",self.model.predict(self.instance)[0])
        self.neighbourhood_targets = self.model.predict(self.final_neighbourhood)
        return self.final_neighbourhood, self.neighbourhood_targets

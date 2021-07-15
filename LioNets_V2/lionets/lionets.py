import matplotlib.pyplot as plt
import random
import seaborn as sns
import numpy as np
from math import sqrt, exp, log
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
from utilities.duplicate_detection import double_distance_detector
from keras.callbacks import ModelCheckpoint


class LioNets:
    """Class for interpreting a neural network locally"""
    def __init__(self, predictor, decoder, encoder, train_data, target_scaler=None, feature_names=None, decoder_lower_threshold=0, double_detector=False,  embeddings=False, tk=None, word_apheresis=False):
        """Init function
        Args:
            model: The trained predictor model
            autoencoder: The trained autoencoder
            decoder: The second half of the autoencoder
            encoder: The first half of the autoencoder
            train_data: The training data are necessary to compute their encoded feature's statistics
            feature_names: The selected features. The above networks have been trained with these.
            decoder_lower_threshold: Only for tfidf decoders!
        """
        self.predictor = predictor
        self.decoder = decoder
        self.encoder = encoder
        self.train_data = train_data
        self.target_scaler = target_scaler
        self.feature_names = feature_names
        self.decoder_lower_threshold = decoder_lower_threshold
        self.word_apheresis = word_apheresis
        self.double_detector = double_detector
        self.features_statistics = {}
        self.encoded_training_data = encoder.predict(train_data)
        self._extract_feature_statistics()
        self.embeddings = embeddings #boolean indication that we have embeddings
        self.tk = tk #tokenizer
        self.memory = {}

    def explain_instance(self, new_instance, max_neighbours=None, model=None, random_state=0):
        """Generates the explanation for an instance
        Args:
            new_instance: The instance to explain
            max_neighbours:
            model:
            random_state: 
        Return:
            weights: 
            instance_prediction: 
            local_prediction:
        """
        if len(new_instance.shape) == 2:
            if tuple(list(new_instance.reshape((new_instance.shape[0]*new_instance.shape[1],)))+[max_neighbours]) in self.memory:
                neighbourhood, predictions, distances = self.memory[tuple(list(new_instance.reshape((new_instance.shape[0]*new_instance.shape[1],)))+[max_neighbours])]
            else:
                neighbourhood, predictions, distances = self._get_decoded_neighbourhood(new_instance, max_neighbours, random_state)
                self.memory[tuple(list(new_instance.reshape((new_instance.shape[0]*new_instance.shape[1],)))+[max_neighbours])] = [neighbourhood, predictions, distances]
        else:
            if tuple(list(new_instance)+[max_neighbours]) in self.memory:
                if self.embeddings:
                    neighbourhood, predictions, distances, local_feature_names = self.memory[tuple(tuple(list(new_instance)+[max_neighbours]))]
                else:
                    neighbourhood, predictions, distances = self.memory[tuple(list(new_instance)+[max_neighbours])]
            else:
                if self.embeddings:
                    neighbourhood, predictions, distances, local_feature_names = self._get_decoded_neighbourhood(new_instance, max_neighbours, random_state)
                    self.memory[tuple(list(new_instance)+[max_neighbours])] = [neighbourhood, predictions, distances, local_feature_names]
                else:
                    neighbourhood, predictions, distances = self._get_decoded_neighbourhood(new_instance, max_neighbours, random_state)
                    self.memory[tuple(list(new_instance)+[max_neighbours])] = [neighbourhood, predictions, distances]
                
        instance_prediction = predictions[-1]

        #train linear model
        if str(type(model)) == "<class 'keras.engine.training.Model'>":
            checkpoint_name = 'local.hdf5' 
            checkpoint = ModelCheckpoint(checkpoint_name, monitor='loss', verbose = 0, save_best_only = True, mode ='auto')
            
            model.fit(np.array(neighbourhood), np.array(predictions),
                                     epochs=10, batch_size=64, sample_weight=np.array(distances), shuffle=True, 
                                     callbacks=[checkpoint], verbose=0)
            model.load_weights('local.hdf5') # load it
            weights = [i for i in model.layers[-1].get_weights()[0]]

            local_prediction = model.predict(np.array([neighbourhood[-1],neighbourhood[-1]]))[0]
            if self.embeddings:
                return weights, instance_prediction, local_prediction, local_feature_names, neighbourhood[-1]
            else:
                return weights, instance_prediction, local_prediction
        else:
            
            #if not 1D representation reshape
            if len(new_instance.shape) == 2: #only for 1 or 2 dimension
                one_dimension_size = new_instance.shape[0] * new_instance.shape[1]
                neighbourhood = neighbourhood.reshape((len(neighbourhood),one_dimension_size))

            linear_model = self._fit_linear_model(neighbourhood, predictions, distances, model)
            weights = linear_model.coef_

            local_prediction = linear_model.predict([neighbourhood[-1]])[0]
            if self.embeddings:
                return weights, instance_prediction, local_prediction, local_feature_names, neighbourhood[-1]
            else:
                return weights, instance_prediction, local_prediction

    def _get_decoded_neighbourhood(self, instance, max_neighbours=None, random_state=0):
        """Returns
        """
        encoded_instance = self.encoder.predict(np.array([instance]))[0]
        #Add an try catch if encoded feature has not 1D representation
        if max_neighbours is None:
            max_neighbours = len(encoded_instance)*4
            if max_neighbours < 1000:
                max_neighbours = 1000
        encoded_neighbourhood = self._neighbourhood_generation(encoded_instance, max_neighbours, random_state)
        decoded_neighbourhood = self.decoder.predict(np.array(encoded_neighbourhood))
        
        if self.embeddings:
            a_n = []
            for dn in decoded_neighbourhood:
                temp_ind = []
                for j in dn:
                    temp_ind.append(np.argmax(j))
                a_n.append(temp_ind)
                
            if self.word_apheresis:    
                for i in set(instance):
                    temp1 = []
                    temp2 = []
                    for l in range(len(instance)):
                        if instance[l] == i:
                            temp1.append(0)
                            temp2.append(1)
                        else:
                            temp1.append(instance[l])
                            temp2.append(instance[l])
                    a_n.append(temp1)
                    a_n.append(temp2)
                
            decoded_neighbourhood = a_n
          
        
        if self.double_detector:
            decoded_neighbourhood = double_distance_detector(decoded_neighbourhood) #Check for duplicates
            
        if self.embeddings:
            a_n_s = []
            for sent in decoded_neighbourhood:
                sentence = ""
                for word in sent:
                    if word != 0 and word != 1:
                        sentence = sentence + self.tk.index_word[word] + " "
                    elif word == 1:
                        sentence = sentence + "UKN "
                a_n_s.append(sentence) 
                
            a_n = [k for k in decoded_neighbourhood]
            original_sentence = ""
            for j in instance:
                if j != 0 and j != 1 :
                    original_sentence = original_sentence + self.tk.index_word[j] + " "
                elif j == 1:
                    original_sentence = original_sentence + 'UKN '
            for o in range(1):
                a_n.append(instance)
                a_n_s.append(original_sentence)
            
            temp_distances = euclidean_distances([encoded_instance],self.encoder.predict(np.array(a_n)))[0]
            distances = []
            dimensions = len(encoded_instance)
            if dimensions < 100:
                dimensions = 100
            for distance in temp_distances:
                distances.append(exp(-(distance*log(dimensions)/2))*log(dimensions))
            predictions = self.predictor.predict(np.array(a_n))
            temp_vec = CountVectorizer().fit(a_n_s)
            neighbourhood = temp_vec.transform(a_n_s)
            neighbourhood = neighbourhood.toarray()
            return neighbourhood, predictions, distances, temp_vec.get_feature_names()
        else:
            temp_neighbourhood = [neighbour for neighbour in decoded_neighbourhood]
            neighbourhood = np.array(temp_neighbourhood)

            if self.decoder_lower_threshold != 0: #TFIDF vectors:
                for neighbour in range(len(neighbourhood)):
                    for feature in range(len(neighbourhood[neighbour])):
                        if neighbourhood[neighbour][feature] <= self.decoder_lower_threshold:
                            neighbourhood[neighbour][feature] = 0
                if self.word_apheresis:
                    t_neighbourhood = list(temp_neighbourhood)
                    for ii in range(len(instance)):
                        temp = instance.copy()
                        temp2 = instance.copy()
                        if temp[ii] != 0:
                            temp[ii] = 0
                            temp2[ii] = 1
                        t_neighbourhood.append(temp)
                        t_neighbourhood.append(temp2)
                    neighbourhood = np.array(t_neighbourhood)

            temp_neighbourhood = [neighbour for neighbour in neighbourhood]
            temp_neighbourhood.append(instance)
            neighbourhood = np.array(temp_neighbourhood)
            
            temp_distances = euclidean_distances([encoded_instance],self.encoder.predict(neighbourhood))[0]
            distances = []
            dimensions = len(encoded_instance)
            if dimensions < 100:
                dimensions = 100
            for distance in temp_distances:
                distances.append(exp(-(distance*log(dimensions)/2))*log(dimensions))

            predictions = self.predictor.predict(neighbourhood)
            if self.target_scaler is not None:
                predictions = self.target_scaler.inverse_transform(predictions)
            predictions = [prediction[0] for prediction in predictions]

        return neighbourhood, predictions, distances



    def _extract_feature_statistics(self):
        encoded_dim = len(self.encoded_training_data[0])
        for feature in range(encoded_dim):
            self.features_statistics[feature] = []
        for feature in range(encoded_dim):
            self.features_statistics[feature].append(
                self.encoded_training_data[:, feature:feature + 1].min())
            self.features_statistics[feature].append(
                self.encoded_training_data[:, feature:feature + 1].max())
            self.features_statistics[feature].append(
                self.encoded_training_data[:, feature:feature + 1].mean())
            self.features_statistics[feature].append(
                self.encoded_training_data[:, feature:feature + 1].std())

    def _neighbourhood_generation(self, encoded_instance, max_neighbours, random_state=0):
        """Generates the neighbourhood of an instance
        Args:
            encoded_instance: The instance to generate neighbours
            max_neighbours: Maximum number of neighbours
            random_state: A seed for stable neighbours generation
        Return:
            neighbours: The generated neighbours
        """
        encoded_dim = len(encoded_instance)
        neighbours = []
        neighbours.append(encoded_instance)
        for feature in range(encoded_dim): #Adding neighbours different by one element
            value = encoded_instance[feature]
            neighbour_copy = encoded_instance.copy()
            
            v1 = self._determine_feature_change(value, feature, random_state=random_state)
            neighbour_copy[feature] = v1
            neighbours.append(neighbour_copy)
            
            v2 = self._determine_feature_change(value, feature, smaller=True, random_state=random_state)
            if v1 != v2:
                neighbour_copy[feature] = v2
                neighbours.append(neighbour_copy)
            
            v3 = self._determine_feature_change(value, feature, bigger=True, random_state=random_state)
            if v3 != v1 and v3 != v2:
                neighbour_copy[feature] = v3
                neighbours.append(neighbour_copy)
        while len(neighbours) < max_neighbours:
            neighbour_copy = encoded_instance.copy()
            np.random.seed(abs(int((neighbour_copy.sum()**2)*100)+len(neighbours))+random_state)
            for f_i in np.random.randint(2, size=encoded_dim).nonzero()[0]:
                value_to_change = neighbour_copy[f_i]
                new_value = self._determine_feature_change(value_to_change, f_i, smaller=True, random_state=random_state)
                neighbour_copy[f_i] = new_value
            neighbours.append(neighbour_copy)
        neighbours = neighbours[:max_neighbours]
        return neighbours

    def _determine_feature_change(self, value, feature, smaller=False, bigger=False, random_state=0):
        """Determines the new value for a specific feature based on gaussian noise
        Args:
            value: Current value of encoded feature
            feature: The feature itself, in order to find its statistics
            smaller: If we want to draw a smaller gaussian noise, and alter a little the original value
            bigger: If we want to draw a bigger gaussian noise, and alter a lot the original value
            random_state: A seed for stable value generation
        Return:
            new_value: The proposed new value
        """
        min_ = self.features_statistics[feature][0]
        max_ = self.features_statistics[feature][1]
        mean_ = self.features_statistics[feature][2]
        std_ = self.features_statistics[feature][3]
        if smaller:
            std_ = std_ / 2
        elif bigger:
            std_ = std_ * 2
        np.random.seed(abs(int((value**2)*100)+int((mean_**2)*100)+int((std_**2)*100)+random_state))
        new_value = np.random.normal(mean_,std_,1)[0] #Gaussian Noise
        if (value + new_value) < self.features_statistics[feature][0]:
            new_value = self.features_statistics[feature][0]
        elif (value + new_value) > self.features_statistics[feature][1]:
            new_value = self.features_statistics[feature][1]
        else:
            new_value = value + new_value
        return new_value

    def _fit_linear_model(self, X,y,distances, model=None):
        """Determines the new value for a specific feature based on gaussian noise
        Args:
            X:
            y:
            distances:
        Return:
            best_model: 
        """
        if model is None:
            best_model = None
            best_score = 10000
            alphas = [0.1, 1, 10, 100, 1000]
            for a in alphas:
                #temp_model = SGDRegressor(alpha=a, penalty='elasticnet').fit(X,y,sample_weight=distances)
                #temp_performance = abs(mean_squared_error(y,temp_model.predict(X)))
                #if best_score > temp_performance:
                #    best_score = temp_performance
                #    best_model = temp_model
                temp_model = Ridge(alpha=a, fit_intercept=True).fit(X,y,distances)
                temp_performance = abs(mean_squared_error(y,temp_model.predict(X)))
                if best_score > temp_performance:
                    best_score = temp_performance
                    best_model = temp_model
            return best_model
        else:
            return model.fit(X,y,distances)


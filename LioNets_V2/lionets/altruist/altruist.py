import numpy as np
import math
import os

class Altruist:
    """
    Altruist:  A tool for providing more truthful interpretations, as well as a tool for selection and benchmarking.
    Altruist works with every machine learning model which provides predictions in the form of continuous values (eg. probabilities).
    It uses feature importance techniques like LIME, SHAP, etc.
    ...
    
    Methods
    -------
    find_untruthful_features(instance)
        It generates the interpretations for this instance's prediction. Then, identifies the untruthful features in each interpretation. In the process
        saves any counterfactual information found in the process. It returns the list of untruthful features of each interpretation.
    explain_why(instance, fi, truthful_only=False)
        It runs the exact same pipeline as the previous pipeline, but only for a selected feature importance technique. It generates the arguments which explain how the untruthful features occured, and why the interpretation is not trusted.
    """

    def __init__ (self, model, training_data, fi_technique, feature_names, cbi_features, nn = None, reshape=None, embeddings=False, noText=False):
        """
        Parameters
        ----------
            model: ml model
                The machine learning model which must provide either continuous output (regression problems), or output in the form of probabilities (classification problems).
            training_data: numpy array
                The data that the machine learning have been trained on
            fi_technique: function or list of functions
                The interpretation(s) technique(s) provided by the system designer / user
            feature_names: list
                The names of the features
            cbi_features: list
                Categorical, binary or integer features. This will help to choose the alterations of a feature's value.
        """
        self.model = model
        self.training_data = training_data
        self.fi_technique = fi_technique #function when 1, list of function when 2+
        self.feature_names = feature_names
        self.cbi_features = cbi_features
        self.nn = nn
        self.multiple_fis = False
        if type(self.fi_technique) is list:
            self.multiple_fis = True
        if self.multiple_fis:
            self.fis = len(fi_technique)
        else:
            self.fis = 1
        self.code_names = []
        self.map_feature_names = {}
        temp_count = 1
        for featue in feature_names:
            name = "F" + str(temp_count)
            self.map_feature_names[name] = featue
            self.code_names.append(name)
            temp_count = temp_count + 1
        self.number_of_features = len(feature_names)
        self.features_statistics = {}
        self.prolog = True
        self.noText = noText
        self.reshape = reshape
        if reshape is not None:
            self.noText = True
        self.embeddings = embeddings
        self._extract_feature_statistics()
        
    def find_untruthful_features(self, instance):
        """
        Parameters
        ----------
            instance: array
                The instance which prediction's interpretation(s) Altruist will investigate.
                
        Returns
        -------
        list
            The untruthful features appearing in the interpretation(s) technique(s)
        list
            A list of counterfactual values, that may change drastically the prediction
        """
        if self.prolog:
            return self._prolog_query(instance)
        else:
            return "Currently not Available without SWIPL"
            
    def explain_why(self, instance, fi, truthful_only=False):
        """
        Parameters
        ----------
            instance: array
                The instance which prediction's interpretation(s) Altruist will investigate
            fi: function
                The preferred feature importance technique to be explained
            truthful_only: bool, optional
                If it should provide explanations for the truthful features only or for all
                
        Returns
        -------
        list
            The untruthful features appearing in the interpretation(s) technique(s)
        list
            A list of counterfactual values, that may change drastically the prediction
        """
        if self.prolog:
            return self._prolog_query_explain(instance, fi, truthful_only)
        else:
            return "Currently not Available without SWIPL"
            
    def _prolog_query(self, instance):
        fi_truthfulness = []
        counter_factuals = []
        for fi in range(1,self.fis+1):
            counter_factuals.append(self._write_pl(instance, fi))
            a = "'altruist/prolog_outputs/temp_"+str(fi)+".pl'"
            b = "'Explanation'"
            untruthful_features = []
            os.system('swipl -g "consult('+a+'), untrusted('+b+')" 2>&1 | tee altruist/prolog_outputs/temp_out.txt')
            tf = open("altruist/prolog_outputs/temp_out.txt","r")
            for x in tf:
                if 'indeed' in x:
                    untruthful_features.append(self.map_feature_names[x.split()[0]])
            tf.close()
            fi_truthfulness.append(untruthful_features)
        return fi_truthfulness, counter_factuals
        
    def _prolog_query_explain(self, instance, fi, truthful_only):
        final = []
        arguments = {}
        for i in range(1,1+len(self.feature_names)):
            arguments[str("F"+str(i))] = []
    
        counter_factuals = self._write_pl(instance, fi+1, explain=True)
        a = "'altruist/prolog_outputs/temp_"+str(fi+1)+"_explain.pl'"
        b = "'Explanation'"
        
        untruthful_features = []
        os.system('swipl -g "consult('+a+'), untrusted('+b+')" 2>&1 | tee altruist/prolog_outputs/temp_out_explain.txt')
        tf = open("altruist/prolog_outputs/temp_out_explain.txt","r")
        for x in tf:
            final.append(x[:-1])
            for i in list(reversed(range(1,1+len(self.feature_names)))):
                if str("F"+str(i)) in x[:-1]:
                    arguments[str("F"+str(i))].append(x[:-1])
                    break
        if len(final) > 1:
            final = final[-2]
            if final == 'trusted("Explanation") is valid':
                nodes_from = []
                nodes_to = []
                args = {}
                args['A1'] = "untrusted('Explanation')"
                nodes_to.append('A1')
                args['A2'] = 'trusted("Explanation") is valid'
                nodes_from.append('A2')
                count = 3
                for i in range(1,len(self.feature_names)+1):
                    key = str("F"+str(i))
                    temp_coun = 0
                    temp_len = len(list(reversed(arguments[key])))
                    if (truthful_only and temp_len == 6) or (not truthful_only):
                        if temp_len == 6:
                            for j in list(reversed(arguments[key])):
                                if temp_coun == 0:
                                    nodes_to.append('A2')
                                    nodes_from.append(str('A'+str(count)))
                                    args[str('A'+str(count))] = j
                                    count = count + 1
                                    temp_coun = temp_coun + 1
                                elif temp_coun == 4:
                                    nodes_to.append(str('A'+str(count-3)))
                                    nodes_from.append(str('A'+str(count)))
                                    args[str('A'+str(count))] = j
                                    count = count + 1
                                    temp_coun = temp_coun + 1
                                elif temp_coun == 5:
                                    nodes_to.append(str('A'+str(count-1)))
                                    nodes_from.append(str('A'+str(count)))
                                    args[str('A'+str(count))] = j
                                    count = count + 1
                                    temp_coun = temp_coun + 1
                                else:
                                    nodes_to.append(str('A'+str(count-1)))
                                    nodes_from.append(str('A'+str(count)))
                                    args[str('A'+str(count))] = j
                                    count = count + 1
                                    temp_coun = temp_coun + 1
                        if temp_len == 5:
                            temp_j = ''
                            for j in list(reversed(arguments[key])):
                                if temp_coun == 0:
                                    nodes_to.append('A2')
                                    nodes_from.append(str('A'+str(count)))
                                    args[str('A'+str(count))] = j
                                    count = count + 1
                                    temp_coun = temp_coun + 1
                                elif temp_coun == 3:
                                    nodes_to.append(str('A'+str(count-2)))
                                    nodes_from.append(str('A'+str(count)))
                                    args[str('A'+str(count))] = j
                                    count = count + 1
                                    temp_coun = temp_coun + 1
                                    temp_j = j
                                elif temp_coun == 4:
                                    if 'raising' in temp_j and 'higher' in j:
                                        cc = 0
                                    else:
                                        cc = 1
                                    nodes_to.append(str('A'+str(count-(1+cc))))
                                    nodes_from.append(str('A'+str(count)))
                                    args[str('A'+str(count))] = j
                                    count = count + 1
                                    temp_coun = temp_coun + 1
                                else:
                                    nodes_to.append(str('A'+str(count-1)))
                                    nodes_from.append(str('A'+str(count)))
                                    args[str('A'+str(count))] = j
                                    count = count + 1
                                    temp_coun = temp_coun + 1
                        elif temp_len == 4:
                            for j in list(reversed(arguments[key])):
                                if temp_coun == 0:
                                    nodes_to.append('A2')
                                    nodes_from.append(str('A'+str(count)))
                                    args[str('A'+str(count))] = j
                                    count = count + 1
                                    temp_coun = temp_coun + 1
                                elif temp_coun == 3 :
                                    nodes_to.append(str('A'+str(count-2)))
                                    nodes_from.append(str('A'+str(count)))
                                    args[str('A'+str(count))] = j
                                    count = count + 1
                                    temp_coun = temp_coun + 1
                                else:
                                    nodes_to.append(str('A'+str(count-1)))
                                    nodes_from.append(str('A'+str(count)))
                                    args[str('A'+str(count))] = j
                                    count = count + 1
                                    temp_coun = temp_coun + 1
                            
        return args, counter_factuals, nodes_to, nodes_from
        
    def _write_pl(self, instance, fi, explain=False):
        predicates, counter_factuals = self._create_pl_file(instance,fi)
        if explain:
            tf = open("altruist/base_b.pl","r")
            f = open("altruist/prolog_outputs/temp_"+str(fi)+"_explain.pl", "w")
        else:
            tf = open("altruist/base_a.pl","r")
            f = open("altruist/prolog_outputs/temp_"+str(fi)+".pl", "w")
        for x in tf:
        	f.write(x)
        for predicate in predicates:
            f.write(predicate+"\n")
        tf.close()
        f.close()
        return counter_factuals
        
    def _create_pl_file(self, instance, fi):
        if self.nn:
            if self.reshape is not None:
                prediction = self.model.predict(np.array([instance.reshape((self.reshape))]))[0] #neural
            else:
                prediction = self.model.predict(np.array([instance]))[0] #neural
        else:
            if self.reshape is not None:
                prediction = self.model.predict([instance.reshape((self.reshape))])[0] #prosoxh ta models pou dinoun shape (2,1)
            else:
                prediction = self.model.predict([instance])[0] #prosoxh ta models pou dinoun shape (2,1)
        if self.multiple_fis:
            feature_importance = self.fi_technique[fi-1](instance,prediction,self.model)#? tha einai hash map {feature: influence} xreiazetai to pred?
        else:
            feature_importance = self.fi_technique(instance,prediction,self.model)#? tha einai hash map {feature: influence} xreiazetai to pred?
        list_of_predicates = []
        list_of_evs = []
        counter_factuals = []
        trusted_query = "trusted('Explanation') :- "
        for feature in range(self.number_of_features): 
            if instance[feature] != 0 or self.noText:#Here put and
                feature_name = self.code_names[feature]
                importance = feature_importance[feature]
                if importance > 0:
                    list_of_predicates.append("feature_importance('"+feature_name+"', 'Positive').")
                elif importance < 0:
                    list_of_predicates.append("feature_importance('"+feature_name+"', 'Negative').")
                else:
                    list_of_predicates.append("feature_importance('"+feature_name+"', 'Neutral').")
                trusted_query = trusted_query + "not(untruthful('"+feature_name+"')), writeln('"+feature_name+" is untruthful'), "
                eval = self._evaluated(feature,importance,instance)
                [list_of_evs.append(i) for i in eval[0]]
                [counter_factuals.append(i) for i in eval[1]]
        [list_of_predicates.append(i) for i in list_of_evs]
        list_of_predicates.append(trusted_query[:-2]+'.')
        return list_of_predicates, counter_factuals
    
    def _evaluated(self, feature, importance, instance):
        list_of_evaluated = []
        counter_factuals = []
        flag_max = False
        flag_min = False
        feature_name = self.code_names[feature]
        temp_instance_i = instance.copy()
        temp_instance_d = instance.copy()
        
        if self.reshape is not None:
            for i in range(50):
                to_be_evaluated = self._determine_feature_change(instance[feature+i*self.reshape[1]],feature)
                if instance[feature+i*self.reshape[1]] == to_be_evaluated[0]:
                    flag_max = True
                if instance[feature+i*self.reshape[1]] == to_be_evaluated[1]:
                    flag_min = True
                temp_instance_i[feature+i*self.reshape[1]] = to_be_evaluated[0]
                temp_instance_d[feature+i*self.reshape[1]] = to_be_evaluated[1]
            temp_instance_i = temp_instance_i.reshape((self.reshape))
            temp_instance_d = temp_instance_d.reshape((self.reshape))
            instance = instance.reshape((self.reshape))
            
        else:
            to_be_evaluated = self._determine_feature_change(instance[feature],feature,True)
            if instance[feature] == to_be_evaluated[0]:
                flag_max = True
            if instance[feature] == to_be_evaluated[1]:
                flag_min = True
            temp_instance_i[feature] = to_be_evaluated[0]
            temp_instance_d[feature] = to_be_evaluated[1]
        if self.nn:
            probabilities = self.model.predict(np.array([instance,temp_instance_i,temp_instance_d])) #neural
            probabilities = [probabilities[0][0],probabilities[1][0],probabilities[2][0]]
        else:
            probabilities = self.model.predict_proba([instance,temp_instance_i,temp_instance_d]) #Check here if it has problem predictinh like before, maybe create _predict do it to be able to addapt in regression as well
            probabilities = [probabilities[0][0],probabilities[1][0],probabilities[2][0]]
        if (probabilities[0] < 0.5 and probabilities[1] >= 0.5) or (probabilities[0] >= 0.5 and probabilities[1] < 0.5):
            counter_factuals.append([feature,to_be_evaluated[0]])
        if (probabilities[0] < 0.5 and probabilities[2] >= 0.5) or (probabilities[0] >= 0.5 and probabilities[2] < 0.5):
            counter_factuals.append([feature,to_be_evaluated[1]])
        if importance > 0:
            if flag_max or probabilities[0] < probabilities[1]:
                list_of_evaluated.append("evaluated('"+feature_name+"','Positive','+','Increase').")
            if flag_min or probabilities[0] > probabilities[2]:
                list_of_evaluated.append("evaluated('"+feature_name+"','Positive','-','Decrease').")
        elif importance < 0:
            if flag_max or probabilities[0] > probabilities[1]:
                list_of_evaluated.append("evaluated('"+feature_name+"','Negative','+','Decrease').")
            if flag_min or probabilities[0] < probabilities[2]:
                list_of_evaluated.append("evaluated('"+feature_name+"','Negative','-','Increase').")
        else:
            if flag_max or probabilities[0] == probabilities[1] or abs(probabilities[0] - probabilities[1]) < 0.01:
                list_of_evaluated.append("evaluated('"+feature_name+"','Neutral','+','Stable').")
            if flag_min or probabilities[0] == probabilities[2] or abs(probabilities[0] - probabilities[2]) < 0.01:
                list_of_evaluated.append("evaluated('"+feature_name+"','Neutral','-','Stable').")
        return list_of_evaluated, counter_factuals
    #def _feature_statistics():
        
    def _extract_feature_statistics(self):
        if self.reshape is not None:
            number_of_features = self.reshape[0]*self.reshape[1]
            for feature in range(number_of_features):
                self.features_statistics[feature] = []
            for feature in range(self.number_of_features):
                self.features_statistics[feature].append(
                    self.training_data[:, feature:feature + 1].min())
                self.features_statistics[feature].append(
                    self.training_data[:, feature:feature + 1].max())
                self.features_statistics[feature].append(
                    self.training_data[:, feature:feature + 1].mean())
                self.features_statistics[feature].append(
                    self.training_data[:, feature:feature + 1].std())
        else:
            for feature in range(self.number_of_features):
                self.features_statistics[feature] = []
            for feature in range(self.number_of_features):
                self.features_statistics[feature].append(
                    self.training_data[:, feature:feature + 1].min())
                self.features_statistics[feature].append(
                    self.training_data[:, feature:feature + 1].max())
                self.features_statistics[feature].append(
                    self.training_data[:, feature:feature + 1].mean())
                self.features_statistics[feature].append(
                    self.training_data[:, feature:feature + 1].std())
        
        
    def _determine_feature_change(self, value, feature, random_state=0):
        #Please move them to 0,1
        min_ = self.features_statistics[feature][0]
        max_ = self.features_statistics[feature][1]
        mean_ = self.features_statistics[feature][2]
        std_ = self.features_statistics[feature][3]

        np.random.seed(abs(int((value**2)*100)+int((mean_**2)*100)+int((std_**2)*100)+random_state))
        #np.random.seed(abs(int((value**2)*10)+int((mean_**2)*10)+int((std_**2)*10)+random_state))
       
        noise = abs(mean_ - np.random.normal(mean_,std_,1)[0]) #Gaussian Noise/
        new_value = value + noise
        new_value_op = value - noise
         
        if new_value > max_:
            new_value = max_
        if new_value_op < min_:
            new_value_op = min_
        if self.cbi_features is not None and feature in self.cbi_features:
            if self.cbi_features[feature] == 1:
                new_value = math.ceil(new_value)
                new_value_op = math.floor(new_value_op)
        if self.embeddings:
            new_value = value
            new_value_op = 0
        return new_value, new_value_op

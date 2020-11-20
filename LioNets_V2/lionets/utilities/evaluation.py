from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, balanced_accuracy_score, accuracy_score
import numpy as np
from math import sqrt, exp, log

class Evaluation():
    def __init__(self, predict, interpretation_technique, transformation, toarray=False, lime=False):
        self.predict = predict
        self.interpretation_technique = interpretation_technique
        if transformation == None:
            self.transformation = lambda x: x
        else:
            self.transformation = transformation
        self.toarray=toarray
        self.lime=lime
        self.ti = 0
        self.non_zero = 0
        self.full_results = []
    
    def fidelity(self, data, interpretation_techniques, class_n=0):
        skipped = 0
        fid = []
        for i in range(len(interpretation_techniques)):
            fid.append([])
        tdata = self.transformation(data)
        predictions = self.predict(tdata)
        for instance in data:
            cit = 0
            for it in interpretation_techniques:
                local_prediction = it(instance)
                fid[cit].append(local_prediction)
                cit = cit + 1
        final_fidelity = []
        for fi in fid:
            tf = []
            tf.append(mean_absolute_error(predictions,fi))
            final_fidelity.append(tf)
        return final_fidelity
    
    def non_zero_weights(self, data, interpretation_techniques):
        non_zero = []

        for i in range(len(interpretation_techniques)):
            non_zero.append([])
        for instance in data:
            cit = 0
            for it in interpretation_techniques:
                weights = it(instance)
                non_zero[cit].append(len(weights.nonzero()[0]))
                cit = cit + 1
        final_non_zero = []
        for nz in non_zero:
            tnz = [np.array(nz).mean()]
            final_non_zero.append(tnz)
        return final_non_zero
         
    def robustness(self, data, interpretation_techniques, voc=None,reshape=None): #chi the technique from which we will pick the change
        skipped = 0
        robustness = []
        transformed_data = self.transformation(data)
        for ind in range(len(data)):
            instance = data[ind].copy()
            interpretations = []
            for i in interpretation_techniques:
                interpretations.append(i(instance))
            index = -1
            c = 0
            abs_min_inf = 10000
            for f in interpretations[0]:
                if abs(f) < abs_min_inf and abs(f) !=0 :
                    abs_min_inf = abs(f)
                    index = c
                c = c + 1
            if voc is not None:
                word = voc[index]
                instance = instance.replace(word,'')
            if reshape is not None:
                t_instance = instance.reshape((reshape[0]))
                if t_instance[index] != 0.1:
                    t_instance[index] = 0.1
                else:
                    t_instance[index] = 1.1
                instance = t_instance.reshape((reshape[1]))

            interpretations_c = []
            def ndif(a,b):
                minv = a.min() if a.min() < b.min() else b.min()
                maxv = a.max() if a.max() > b.max() else b.max()
                t_a = []
                t_b = []
                for r in range(len(a)):
                    t_a.append((a[r]-minv)/(maxv-minv))
                    t_b.append((b[r]-minv)/(maxv-minv))
                a = np.array(t_a)
                b = np.array(t_b)
                return np.average(np.abs((a - b)))
            interpretations_dif = []
            s = 0
            for i in interpretation_techniques:
                interpretations_c.append(i(instance))
                interpretations_dif.append(ndif(np.array(interpretations[s]),np.array(interpretations_c[s])))
                s = s + 1
            robustness.append(interpretations_dif)
            #robustness.append([interpretations,interpretations_c])
        robustness = np.array(robustness)
        f_robustness = []
        for ind in range(len(interpretation_techniques)):
            f_robustness.append(np.average(robustness[:,ind]))
        return f_robustness
    
    def robustness_embeddings(self, data, interpretation_techniques, voc=None,reshape=None): #chi the technique from which we will pick the change
        skipped = 0
        robustness = []
        transformed_data = self.transformation(data)
        for ind in range(len(data)):
            instance = data[ind].copy()
            interpretations = []
            for i in interpretation_techniques:
                interpretations.append(i(instance))
            instance[instance.nonzero()[0][0]] = 0
            
            interpretations_c = []
            def ndif(a,b):
                minv = a.min() if a.min() < b.min() else b.min()
                maxv = a.max() if a.max() > b.max() else b.max()
                t_a = []
                t_b = []
                for r in range(len(a)):
                    t_a.append((a[r]-minv)/(maxv-minv))
                    t_b.append((b[r]-minv)/(maxv-minv))
                a = np.array(t_a)
                b = np.array(t_b)
                return np.average(np.abs((a - b)))
            interpretations_dif = []
            s = 0
            for i in interpretation_techniques:
                interpretations_c.append(i(instance))
                interpretations_dif.append(ndif(np.array(interpretations[s]),np.array(interpretations_c[s])))
                s = s + 1
            robustness.append(interpretations_dif)
            #robustness.append([interpretations,interpretations_c])
        robustness = np.array(robustness)
        f_robustness = []
        for ind in range(len(interpretation_techniques)):
            f_robustness.append(np.average(robustness[:,ind]))
        return f_robustness
    
   
    def truthful_influence(self, data, level=1, class_n=0, fidelity=False):
        skipped = 0
        ti = 0
        non_zero = 0
        if fidelity:
            fid = []
        for instance in range(len(data)):
            transformed_instance = self.transformation(data[instance:instance+1])
            classification = self.predict(transformed_instance)[class_n]
            if fidelity:
                if self.toarray:
                    interpretation = self.interpretation_technique(transformed_instance.toarray())[0]
                elif self.lime:
                    interpretation = self.interpretation_technique(data[instance])
                else:
                    interpretation = self.interpretation_technique(transformed_instance)[0]
            else:
                if self.toarray:
                    interpretation = self.interpretation_technique(transformed_instance.toarray())[0]
                elif self.lime: 
                    interpretation = self.interpretation_technique(data[instance])
                else:
                    interpretation = self.interpretation_technique(transformed_instance)[0]
            if len(interpretation.nonzero()[0]) < 2:
                skipped = skipped + 1
            else:
                non_zero = non_zero + len(interpretation.nonzero()[0])
                if fidelity:
                    if self.toarray:
                        fidelity.append(self.interpretation_technique(transformed_instance.toarray())[1])
                    else:
                        fidelity.append(self.interpretation_technique(transformed_instance)[1])
                #In the future this should be able to compute ti for level > 1
                temp_inst = transformed_instance.copy()[0]
                if self.toarray or self.lime:
                    temp_inst = transformed_instance.toarray().copy()[0]
                #print(temp_inst)
                if classification > 0.5:
                    max_influence = interpretation.max()
                    ind = np.argmax(interpretation)
                    temp_inst[ind] = 0
                    new_classification = self.predict(np.array([temp_inst]))[class_n]
                    if classification > new_classification:
                        ti = ti + 1
                    elif new_classification == classification and max_influence == 0:
                        ti = ti + 1
                else:
                    min_influence = interpretation.min()
                    ind = np.argmin(interpretation)
                    temp_inst[ind] = 0
                    new_classification = self.predict(np.array([temp_inst]))[class_n]
                    if classification < new_classification:
                        ti = ti + 1
                    elif new_classification == classification and min_influence == 0:
                        ti = ti + 1
        result = []
        if (len(data)-skipped) > 0:
            result.append(non_zero/(len(data)-skipped))
            result.append(ti/(len(data)-skipped))
        if fidelity:
            fid1 = [i[0] for i in fid]
            fid2 = [i[1] for i in fid]
            self.fidelity_mean_absolute = mean_absolute_error(fid1,fid2)
            self.fidelity_squared_error = sqrt(mean_squared_error(fid1,fid2))
            self.fidelity_r2 = r2_score(fid1,fid2)
            result.append(self.fidelity_mean_absolute)
            result.append(self.fidelity_squared_error)
            result.append(self.fidelity_r2)
        return result



        
    def upper_text(self, instance, interpretation):
        return None
    def lower_text(self, instance, interpretation):
        return None

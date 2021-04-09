# full survey from forest_walker objects
    def full_survey(self
        , instances
        , labels):

        self.instances = instances
        self.labels = labels
        self.n_instances = instances.shape[0]

        if labels is not None:
            if len(labels) != self.n_instances:
                raise ValueError("number of labels and instances does not match")

        # base counts for all trees
        self.root_child_lower = {}

        # walk through each tree to get the structure
        for t, trees in enumerate(self.forest.estimators_):
            # because gbm does one versus all for multiclass
            if type(self.forest) == GradientBoostingClassifier:
                class_trees = trees
            else:
                class_trees = [trees]
            for ct, ctree in enumerate(class_trees): # this is an individual estimator
                if t == 0:
                    self.root_child_lower[ct] = {'root_features' : np.zeros(len(self.features)),  # set up a 1d feature array to count features appearing as root nodes
                    'child_features' : np.zeros(len(self.features)),
                    'lower_features' : np.zeros(len(self.features))}

                # root, child and lower counting, one time only (first class)
                structure = ctree.tree_
                feature = structure.feature
                children_left = structure.children_left
                children_right = structure.children_right

                self.root_child_lower[ct]['root_features'][feature[0]] += 1
                if children_left[0] >= 0:
                    self.root_child_lower[ct]['child_features'][feature[children_left[0]]] +=1
                if children_right[0] >= 0:
                    self.root_child_lower[ct]['child_features'][feature[children_right[0]]] +=1

                for j, f in enumerate(feature):
                    if j < 3: continue # root and children
                    if f < 0: continue # leaf nodes
                    self.root_child_lower[ct]['lower_features'][f] += 1
        self.tree_outputs = {}

        # walk through each tree
        self.n_trees = len(self.forest.estimators_)
        for t, trees in enumerate(self.forest.estimators_):
            # because gbm does one versus all for multiclass
            if type(self.forest) == GradientBoostingClassifier:
                class_trees = trees
            else:
                class_trees = [trees]
            for ct, ctree in enumerate(class_trees): # this is an individual estimator
                if t == 0: # initialise the dictionary
                    self.tree_outputs[ct] = {'feature_depth' : np.full((self.n_instances, self.n_trees, self.n_features), np.nan), # set up a 1d feature array for counting
                    'tree_predictions' : np.full((self.n_instances, self.n_trees), np.nan),
                    'tree_pred_labels' : np.full((self.n_instances, self.n_trees), np.nan),
                    'tree_performance' : np.full((self.n_instances, self.n_trees), np.nan),
                    'path_lengths' : np.zeros((self.n_instances, self.n_trees))
                    }

                # get the feature vector out of the tree object
                feature = ctree.tree_.feature

                self.tree_outputs[ct]['tree_predictions'][:, t] = ctree.predict(self.instances)
                if type(self.forest) == GradientBoostingClassifier:
                    tpr = np.sign(self.tree_outputs[ct]['tree_predictions'][:, t])
                    tpr[tpr < 0] = 0
                    self.tree_outputs[ct]['tree_pred_labels'][:, t] = tpr
                else:
                    self.tree_outputs[ct]['tree_pred_labels'][:, t] = self.tree_outputs[ct]['tree_predictions'][:, t]
                self.tree_outputs[ct]['tree_performance'][:, t] = self.tree_outputs[ct]['tree_pred_labels'][:, t] == self.labels

                # extract path and get path lengths
                path = ctree.decision_path(self.instances).indices
                paths_begin = np.where(path == 0)
                paths_end = np.append(np.where(path == 0)[0][1:], len(path))
                self.tree_outputs[ct]['path_lengths'][:, t] = paths_end - paths_begin

                depth = 0
                instance = -1
                for p in path:
                    if feature[p] < 0: # leaf node
                        # TO DO: what's in a leaf node
                        continue
                    if p == 0: # root node
                        instance += 1 # a new instance
                        depth = 0 # a new path
                    else:
                        depth += 1 # same instance, descends tree one more node
                    self.tree_outputs[ct]['feature_depth'][instance][t][feature[p]] = depth



# stats from forest_walker object
    def forest_stats_by_label(self, label = None):
        if label is None:
            idx = Series([True] * self.n_instances) # it's easier if has the same type as the labels
            label = 'all_classes'
        else:
            idx = self.labels == label
        idx = idx.values

        n_instances_lab = sum(idx) # number of instances having the current label
        if n_instances_lab == 0: return

        # object to hold all the statistics
        statistics = {}
        statistics['n_trees'] = self.n_trees
        statistics['n_instances'] = n_instances_lab

        # get a copy of the arrays, containing only the required instances
        feature_depth_lab = self.feature_depth[idx]
        path_lengths_lab = self.path_lengths[idx]
        tree_performance_lab = self.tree_performance[idx]

        # gather statistics from the feature_depth array, for each class label
        # shape is instances, trees, features, so [:,:,fd]
        depth_counts = [np.unique(feature_depth_lab[:,:,fd][~np.isnan(feature_depth_lab[:,:,fd])], return_counts = True) for fd in range(self.n_features)]

        # number of times each feature node was visited
        statistics['n_node_traversals'] = np.array([np.nansum(dcz[1]) for dcz in depth_counts], dtype=np.float32)
        # number of times feature was a root node (depth == 0)
        statistics['n_root_traversals'] = np.array([depth_counts[dc][1][np.where(depth_counts[dc][0] == 0)][0] if depth_counts[dc][1][np.where(depth_counts[dc][0] == 0)] else 0 for dc in range(len(depth_counts))], dtype=np.float32)
        # number of times feature was a root-child (depth == 1)
        statistics['n_child_traversals'] = np.array([depth_counts[dc][1][np.where(depth_counts[dc][0] == 1)][0] if depth_counts[dc][1][np.where(depth_counts[dc][0] == 1)] else 0 for dc in range(len(depth_counts))], dtype=np.float32)
        # number of times feature was a lower node (depth > 1)
        statistics['n_lower_traversals'] = np.array([np.nansum(depth_counts[dc][1][np.where(depth_counts[dc][0] > 1)] if any(depth_counts[dc][1][np.where(depth_counts[dc][0] > 1)]) else 0) for dc in range(len(depth_counts))], dtype=np.float32)
        # number of times feature was not a root
        statistics['n_nonroot_traversals'] = statistics['n_node_traversals'] - statistics['n_root_traversals'] # total feature visits - number of times feature was a root

        # number of correct predictions
        statistics['n_correct_preds'] = np.sum(tree_performance_lab) # total number of correct predictions
        statistics['n_path_length'] = np.sum(path_lengths_lab) # total path length accumulated by each feature

        # above measures normalised over all features
        p_ = lambda x : x / np.nansum(x)

        statistics['p_node_traversals'] = p_(statistics['n_node_traversals'])
        statistics['p_root_traversals'] = p_(statistics['n_root_traversals'])
        statistics['p_nonroot_traversals'] = p_(statistics['n_nonroot_traversals'])
        statistics['p_child_traversals'] = p_(statistics['n_child_traversals'])
        statistics['p_lower_traversals'] = p_(statistics['n_lower_traversals'])
        statistics['p_correct_preds'] = np.mean(tree_performance_lab) # accuracy

        statistics['m_node_traversals'] = np.mean(np.sum(~np.isnan(feature_depth_lab), axis = 1), axis = 0) # mean number of times feature appeared over all instances
        statistics['m_root_traversals'] = np.mean(np.sum(feature_depth_lab == 0, axis = 1), axis = 0) # mean number of times feature appeared as a root node, over all instances
        statistics['m_nonroot_traversals'] = np.mean(np.sum(np.nan_to_num(feature_depth_lab) > 0, axis = 1), axis = 0)
        statistics['m_child_traversals'] = np.mean(np.sum(np.nan_to_num(feature_depth_lab) == 1, axis = 1), axis = 0)
        statistics['m_lower_traversals'] = np.mean(np.sum(np.nan_to_num(feature_depth_lab) > 1, axis = 1), axis = 0)
        statistics['m_feature_depth'] = np.mean(np.nanmean(feature_depth_lab, axis = 1), axis = 0) # mean depth of each feature when it appears
        statistics['m_path_length'] = np.mean(np.nanmean(path_lengths_lab, axis = 1), axis = 0) # mean path length of each instance in the forest
        statistics['m_correct_preds'] = np.mean(np.mean(tree_performance_lab, axis = 1)) # mean prop. of trees voting correctly per instance

        if n_instances_lab > 1: # can't compute these on just one example
            statistics['sd_node_traversals'] = np.std(np.sum(~np.isnan(feature_depth_lab), axis = 1), axis = 0, ddof = 1) # sd of number of times... over all instances and trees
            statistics['sd_root_traversals'] = np.std(np.sum(feature_depth_lab == 0, axis = 1), axis = 0, ddof = 1) # sd of number of times feature appeared as a root node, over all instances
            statistics['sd_nonroot_traversals'] = np.std(np.sum(np.nan_to_num(feature_depth_lab) > 0, axis = 1), axis = 0, ddof = 1) # sd of number of times feature appeared as a nonroot node, over all instances
            statistics['sd_child_traversals'] = np.std(np.sum(np.nan_to_num(feature_depth_lab) == 1, axis = 1), axis = 0, ddof = 1)
            statistics['sd_lower_traversals'] = np.std(np.sum(np.nan_to_num(feature_depth_lab) > 1, axis = 1), axis = 0, ddof = 1)
            statistics['sd_feature_depth'] = np.std(np.nanmean(feature_depth_lab, axis = 1), axis = 0, ddof = 1) # sd depth of each feature when it appears
            statistics['sd_path_length'] = np.std(np.nanmean(path_lengths_lab, axis = 1), axis = 0, ddof = 1)
            statistics['sd_correct_preds'] = np.std(np.mean(tree_performance_lab, axis = 1), ddof = 1) # std prop. of trees voting correctly per instance
            statistics['se_node_traversals'] = sem(np.sum(~np.isnan(feature_depth_lab), axis = 1), axis = 0, ddof = 1, nan_policy = 'omit') # se of mean number of times feature appeared over all instances
            statistics['se_root_traversals'] = sem(np.sum(feature_depth_lab == 0, axis = 1), axis = 0, ddof = 1, nan_policy = 'omit') # se of mean of number of times feature appeared as a root node, over all instances
            statistics['se_nonroot_traversals'] = sem(np.sum(np.nan_to_num(feature_depth_lab) > 0, axis = 1), axis = 0, ddof = 1, nan_policy = 'omit') # sd of number of times feature appeared as a nonroot node, over all instances
            statistics['se_child_traversals'] = sem(np.sum(np.nan_to_num(feature_depth_lab) == 1, axis = 1), axis = 0, ddof = 1, nan_policy = 'omit')
            statistics['se_lower_traversals'] = sem(np.sum(np.nan_to_num(feature_depth_lab) > 1, axis = 1), axis = 0, ddof = 1, nan_policy = 'omit')
            statistics['se_feature_depth'] = sem(np.nanmean(feature_depth_lab, axis = 1), axis = 0, ddof = 1, nan_policy = 'omit') # se depth of each feature when it appears
            statistics['se_path_length'] = sem(np.nanmean(path_lengths_lab, axis = 1), axis = 0, ddof = 1, nan_policy = 'omit')
            statistics['se_correct_preds'] = sem(np.mean(tree_performance_lab, axis = 1), ddof = 1, nan_policy = 'omit') # se prop. of trees voting correctly per instance
        else:
            statistics['sd_node_traversals'] = np.full(self.n_features, np.nan)
            statistics['sd_root_traversals'] = np.full(self.n_features, np.nan)
            statistics['sd_nonroot_traversals'] = np.full(self.n_features, np.nan)
            statistics['sd_child_traversals'] = np.full(self.n_features, np.nan)
            statistics['sd_lower_traversals'] = np.full(self.n_features, np.nan)
            statistics['sd_feature_depth'] = np.full(self.n_features, np.nan)
            statistics['sd_path_length'] = np.full(self.n_features, np.nan)
            statistics['sd_correct_preds'] = np.full(self.n_features, np.nan)
            statistics['se_node_traversals'] = np.full(self.n_features, np.nan)
            statistics['se_root_traversals'] = np.full(self.n_features, np.nan)
            statistics['se_nonroot_traversals'] = np.full(self.n_features, np.nan)
            statistics['se_child_traversals'] = np.full(self.n_features, np.nan)
            statistics['se_lower_traversals'] = np.full(self.n_features, np.nan)
            statistics['se_feature_depth'] = np.full(self.n_features, np.nan)
            statistics['se_path_length'] = np.full(self.n_features, np.nan)
            statistics['se_correct_preds'] = np.full(self.n_features, np.nan)
        return(statistics)

    def forest_stats(self, class_labels = None):

        statistics = {}

        if class_labels is None:
            class_labels = np.unique(self.labels)
        for cl in class_labels:
            statistics[cl] = self.forest_stats_by_label(cl)

        statistics['all_classes'] = self.forest_stats_by_label()
        return(statistics)

    def major_class_from_paths(self, batch_idx, return_counts=False): # needs to be weighted
        pred_classes = [self.path_detail[batch_idx][p]['pred_class'] for p in range(len(self.path_detail[batch_idx]))]

        unique, counts = np.unique(pred_classes, return_counts=True)

        if return_counts:
            return(unique[np.argmax(counts)], dict(zip(unique, counts)))
        else: return(unique[np.argmax(counts)])

archive = zipfile.ZipFile('CHIRPS/source_datafiles_proprietary/usoc.zip', 'r')
var_labels = archive.read('usoc_codes.csv').decode("utf-8").split('\r\n')
var_labels = [var_labels[i].replace("don't know", 'not given')
              .replace('"yes, but unable to code as name of drug(s) not available."', 'yes but unable to code')
              .replace('cannot see point/height already known/doctor has measurement', 'already known')
              .replace('"height attempted, not obtained"', 'attempted not obtained')
              .replace('too busy/taken too long already/ no time', 'too busy')
              .replace('respondent too ill/frail/tired', 'ill or frail or tired')
              .replace('considered intrusive information', 'intrusive')
              .replace('respondent too anxious/nervous/ shy/embarrassed', 'shy')
              .replace('refused (no other reason given)', 'no reason')
              .replace('"no problems experienced, reliable height measurement obtaine"', 'obtained')
              .replace('"weight attempted, not obtained"', 'attempted not obtained')
              .replace('"refused, inapplicable, other invalid"', 'not applicable')
              .replace('"other please specify"', 'other')
              .replace('cannot see point/weight already known/doctor has measurement', 'already known')
              .replace('too busy/taken long enough already/no time', 'too busy')
              .replace('respondent too anxious/nervous/shy/embarrased', 'shy')
              .replace('"no problems experienced, reliable weight measurement obtaine"', 'obtained')
              .replace('"no problems experienced, reliable waist measurement"', 'obtained')
              .replace('respondent agrees to have waist measured', 'agreed')
              .replace('respondent refuses to have waist measured', 'refused')
              .replace('unable to measure waist for reason other than refusal', 'attempted not obtained')
              .replace('"no problems experienced reliable waist measurement"', 'obtained')
              .replace('problems experienced - waist measurement likely to be reliable', 'reliable')
              .replace('problems experienced - waist measurement likely to be slightly unreliable', 'slightly unreliable')
              .replace('problems experienced - waist measurement likely to be unreliable', 'unreliable')
              .replace('increases measurement (e.g. bulky clothing)', 'bulky clothing increase')
              .replace('decreases measurement (e.g. very tight clothing)', 'tight clothing decrease')
              .replace('measurement not affected', 'unaffected')
              .replace('"yes, agrees"', 'yes')
              .replace('"no refuses"', 'no')
              .replace('unable to measure bp for reason other than refusal', 'attempted not obtained')
              .replace('hairstyle or wig', 'hair or wig')
              .replace('turban or other religious headgear', 'headgear')
              .replace('respondent stooped', 'stooped')
              .replace('respondent would not stand still', 'fidget')
              .replace('respondent wore shoes', 'shoes')
              .replace('"other, please specify"', 'other')
              .replace('Weight and body fat measured', 'both measured')
              .replace('weight only measured', 'weight')
              .replace('weight refused', 'refused')
              .replace('weight not attempted', 'not attempted')
              .replace('stones and pounds', 'imperial')
              .replace('both measurements refused', 'refused')
              .replace('attempted but not obtained', 'attempted not obtained')
              .replace('measurement not attempted', 'not attempted')
              .replace('child (15-22 cm)', 'child')
              .replace('adult (22-32 cm)', 'adult')
              .replace('large adult (32-42 cm)', 'adult')
              .replace('blood pressure measurement attempted not obtained', 'attempted not obtained')
              .replace('blood pressure not attempted', 'not attempted')
              .replace('blood pressure measurement refused', 'refused')
              .replace('unable to take measurement', 'attempted not obtained')
              .replace('respondent has the use of both hands', 'both')
              .replace('respondent is unable to use dominant hand', 'non dominant')
              .replace('respondent is unable to use nondomin hand', 'dominant')
              .replace('respondent is unable to use either hand', 'neither')
              .replace('standing without arm support', 'stnoarm')
              .replace('sitting without arm support', 'sitnoarm')
              .replace('standing with arm support', 'starm')
              .replace('sitting with arm support', 'sitarm')
              .replace('all measures obtained', 'all')
              .replace('some measures obtained', 'some')
              .replace('no measures obtained', 'none')
              .replace('yes but unable to take lung function measurement for reason', 'attempted not obtained')
              .replace('unable to take lung function measurement for reason other th', 'attempted not obtained')
              .replace('"lung function measurement attempted, not obtained"', 'attempted not obtained')
              .replace('lung function not attempted', 'not attempted')
              .replace('lung function measurement refused', 'refused')
              .replace('all 5 technically satisfactory blows obtained', 'five blows')
              .replace('"some blows, but less than 5 technically satisfactory blows o"', 'some blows')
              .replace('"attempted, but no technically satisfactory blows obtained"', 'no blows')
              .replace('other reason why measurements not attempted/refused (specify', 'other')
              .replace('no technically satisfactory blow', 'no blows')
              .replace('at least one technically satisfactory blow', 'some blows')
              .replace('respondent lives with parent or person with legal responsibi', 'lives with pog')
              .replace('does @inot@i live with parent or person with legal responsib', 'does not live with pog')
              .replace('temperature of house too cold', 'amb temp cold')
              .replace('temperature of house too hot', 'amb temp hot')
              .replace('respondent unable to give blood sample for reason other than', 'attempted not obtained')
              .replace('refusal', 'refused')
              .replace('storage consent given', 'agreed')
              .replace('consent refused', 'refused')
              .replace('dna consent given', 'agreed')
              .replace('sample taken on first attempt', 'first attempt')
              .replace('sample taken on second attempt', 'second attempt')
              .replace('both attempts failed', 'attempted not obtained')
              .replace('"first attempt failed, did not make second attempt"', 'attempted not obtained')
              .replace('sensitive to tape/plaster', 'yes')
              .replace('@inot@i sensitive to tape/plaster', 'no')
              .replace('"yes, site was re-checked"', 'yes')
              .replace('"no, site was not re-checked"', 'no')
              .replace('thin tights', 'thin')
              .replace('thick tights / socks', 'thick')
              .replace('respondent fitted with pacemaker', 'yes')
              .replace('no pacemaker', 'no')
              .replace('nurse visit conducted - blood sample sent to lab', 'sent')
              .replace('nurse visit conducted - no blood sample to lab', 'not sent')
              .replace('individual has left household', 'left')
              .replace('contact made at household but not with this respondent', 'contact other')
              .replace('refused by person at appointment booking stage', 'refused')
              .replace('refused by person at visit stage', 'refused')
              .replace('office refused', 'refused')
              .replace('proxy refused', 'refused')
              .replace('"broken appointment, no further contact"', 'refused')
              .replace('ill at home/in hospital/away during f/w period (until mar 20', 'unable')
              .replace('physically / mentally unable', 'unable')
              .replace('other unproductive', 'unable')
              .replace('respondent is pregnant', 'unable')
              .replace('respondent has died', 'died')
              .replace('whole household unproductive (refused/no contact/moved/not c', 'unable')
              .split(',') for i in range(len(var_labels))]
var_labels[1500:]


pred_logproba = confidence_weight(pred_probas, 'log_proba')
print('log_proba')
print(pred_logproba)
pred_logproba = (pred_logproba - np.mean(pred_logproba, axis = 1)[:, np.newaxis]) * (len(meta_data['class_names']) - 1) # this is the SAMME.R formula
print('samme.r formula')
print(pred_logproba)
confidence_weights = np.sum(pred_logproba, axis=0)
print('raw_conf_weights')
print(confidence_weights)
confidence_weights = p_count_corrected([i for i in range(len(meta_data['class_names']))], [i for i in range(len(meta_data['class_names']))], confidence_weights)
print('conf_weights')
print(confidence_weights)

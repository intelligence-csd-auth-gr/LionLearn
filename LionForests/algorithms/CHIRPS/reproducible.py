import csv
import json
import time
import timeit
import random
import numpy as np
import pandas as pd
import multiprocessing as mp
from math import sqrt, ceil
import warnings
import scipy.stats as st
from copy import deepcopy
import CHIRPS.routines as rt
import CHIRPS.structures as strcts
from CHIRPS import o_print
from CHIRPS import config as cfg
from lime import lime_tabular as limtab
from anchor import anchor_tabular as anchtab
from defragTrees import DefragModel
from sklearn.tree import DecisionTreeClassifier

# lore required quite a lot of adaptation
import LORE.lore as lore
import LORE.util as loreutil
# import lore.neighbor_generator as nbgen
import LORE.pyyadt as loreyadt
import pickle as cPickle
from deap import base, creator, tools, algorithms

penalise_bad_prediction = lambda mc, tc, value : value if mc == tc else 0 # for global interp methods

def export_data_splits(datasets, project_dir=None, random_state_splits=123):
    # general data preparation
    for dataset in datasets:
        mydata = dataset(project_dir=project_dir)

        # train test split - one off hard-coded random state.
        # later we will vary the state to generate different forests for e.g. benchmarking
        train_index, test_index = mydata.get_tt_split_idx(random_state=random_state_splits)
        # save to csv to use the same splits in methods that aren't available in Python
        mydata.tt_split(train_index, test_index).to_csv(mydata.get_save_path(),
                                                                encoded_features = mydata.features_enc)
    print('Exported train-test data for ' + str(len(datasets)) + ' datasets.')

def unseen_data_prep(ds_container, n_instances=1, which_split='test'):
    # this will normalise the above parameters to the size of the dataset
    n_instances = rt.n_instance_ceiling(ds_container=ds_container, n_instances=n_instances)

    # this gets the next batch out of the data_split_container according to the required number of instances
    # all formats can be extracted, depending on the requirement
    # unencoded, encoded (sparse matrix is the type returned by scikit), ordinary dense matrix also available
    instances, instances_matrix, instances_enc, instances_enc_matrix, labels = ds_container.get_next(n_instances, which_split='test') # default

    return(instances, instances_matrix, instances_enc, instances_enc_matrix, labels)

# function to manage the whole run and evaluation
def CHIRPS_benchmark(forest, ds_container, meta_data, model, n_instances=100,
                    forest_walk_async=True,
                    explanation_async=True,
                    save_path='', save_sensitivity_path=None,
                    dataset_name='',
                    random_state=123, verbose=True, n_cores=None,
                    **kwargs):

    if n_cores is None:
        n_cores = mp.cpu_count()-4

    if save_sensitivity_path is None:
        save_sensitivity_path=save_path
    # 2. Prepare Unseen Data and Predictions
    o_print('Beginning benchmark for ' + dataset_name + ' data.', verbose)
    o_print('Hyper-parameter settings: ' + str(kwargs), verbose)
    o_print('', verbose)

    o_print('Prepare Unseen Data and Predictions for CHIRPS benchmark', verbose)
    # OPTION 1 - batching (to be implemented in the new code, right now it will do just one batch)
    instances, _, instances_enc, instances_enc_matrix, labels = unseen_data_prep(ds_container,
                                                                                n_instances=n_instances)
    # get predictions
    preds = forest.predict(instances_enc)

    # 3.1 - Extract Tree Prediction Paths
    o_print('Walking forest for ' + str(len(labels)) + ' instances... (please wait)', verbose)

    # wrapper object needs the decision forest itself and the dataset meta data (we have a convenience function for this)
    if model == 'GBM':
        f_walker = strcts.regression_trees_walker(forest = forest, meta_data=meta_data)
    else:
        f_walker = strcts.classification_trees_walker(forest = forest, meta_data=meta_data)

    # set the timer
    eval_start_time = time.asctime( time.localtime(time.time()) )
    forest_walk_start_time = timeit.default_timer()

    # do the walk - returns a batch_paths_container (even for just one instance)
    # requires the X instances in a matrix (dense, ordinary numpy matrix) - this is available in the data_split_container
    f_walker.forest_walk(instances = instances_enc_matrix,
                            labels = preds, # we're explaining the prediction, not the true label!
                            forest_walk_async = forest_walk_async,
                            n_cores = n_cores)

    # stop the timer
    forest_walk_end_time = timeit.default_timer()
    forest_walk_elapsed_time = forest_walk_end_time - forest_walk_start_time
    forest_walk_mean_elapsed_time = forest_walk_elapsed_time/len(labels)

    o_print('Forest Walk with async = ' + str(forest_walk_async), verbose)
    o_print('Forest Walk time elapsed: ' + '{:0.4f}'.format(forest_walk_elapsed_time) + ' seconds', verbose)
    o_print('', verbose)

    # 3.2-3.4 - Freqent pattern mining of paths, Score and sort mined path segments, Merge path segments into one rule
    # get what the model predicts on the training sample
    sample_labels = forest.predict(ds_container.X_train_enc)

    # build CHIRPS and a rule for each instance represented in the batch paths container
    if model == 'GBM':
        # create a GBHIPS container object for the forest path detail
        explanations = strcts.GBHIPS_container(f_walker.path_detail,
                                        forest=forest,
                                        sample_instances=ds_container.X_train_enc, # any representative sample can be used
                                        sample_labels=sample_labels,
                                        meta_data=meta_data,
                                        forest_walk_mean_elapsed_time=forest_walk_mean_elapsed_time)

        o_print('Running GBHIPS on a batch of ' + str(len(labels)) + ' instances... (please wait)', verbose)
        # start a timer
        ce_start_time = timeit.default_timer()
    else:
        # create a CHIRPS container for the forest path detail
        explanations = strcts.CHIRPS_container(f_walker.path_detail,
                                        forest=forest,
                                        sample_instances=ds_container.X_train_enc, # any representative sample can be used
                                        sample_labels=sample_labels,
                                        meta_data=meta_data,
                                        forest_walk_mean_elapsed_time=forest_walk_mean_elapsed_time)

        o_print('Running CHIRPS on a batch of ' + str(len(labels)) + ' instances... (please wait)', verbose)
        # start a timer
        ce_start_time = timeit.default_timer()


    # run the explanation algorithm on all the instance path details
    explanations.run_explanations(target_classes=preds, # we're explaining the prediction, not the true label!
                            explanation_async=explanation_async,
                            random_state=random_state, n_cores=n_cores,
                            **kwargs)

    ce_end_time = timeit.default_timer()
    ce_elapsed_time = ce_end_time - ce_start_time
    o_print('explanation time elapsed: ' + "{:0.4f}".format(ce_elapsed_time) + ' seconds', verbose)
    o_print('explanation with async = ' + str(explanation_async), verbose)
    o_print('', verbose)

    # 4. Evaluating CHIRPS Explanations
    o_print('Evaluating found explanations', verbose)

    results_start_time = timeit.default_timer()

    # iterate over all the test instances (based on the ids in the index)
    # scoring will leave out the specific instance by this id.

    save_results_file = model + '_CHIRPS_rnst_' + str(random_state)
    rt.evaluate_explainers(explanations, ds_container, labels.index,
                                  forest=forest,
                                  meta_data=meta_data,
                                  eval_start_time=eval_start_time,
                                  print_to_screen=False, # set True when running single instances
                                  save_results_path=save_sensitivity_path,
                                  dataset_name=dataset_name,
                                  model=model,
                                  save_results_file=save_results_file,
                                  save_CHIRPS=False)

    results_end_time = timeit.default_timer()
    results_elapsed_time = results_end_time - results_start_time
    o_print('Results saved to ' + save_sensitivity_path + save_results_file, verbose)
    o_print('CHIRPS batch results eval time elapsed: ' + "{:0.4f}".format(results_elapsed_time) + ' seconds', verbose)
    o_print('', verbose)
    # this completes the CHIRPS runs

def Anchors_preproc(ds_container, meta_data):

    ds_cont = deepcopy(ds_container)
    # create the discretise function from LIME tabular
    disc = limtab.QuartileDiscretizer(np.array(ds_cont.X_train),
                                      categorical_features=meta_data['categorical_features'],
                                      feature_names=meta_data['features'])

    # create a copy of the var_dict with updated labels
    var_dict_anch = deepcopy(meta_data['var_dict'])
    for vk in var_dict_anch.keys():
        if var_dict_anch[vk]['data_type'] == 'continuous':
            var_dict_anch[vk]['labels'] = disc.names[var_dict_anch[vk]['order_col']]
            var_dict_anch[vk]['data_type'] = 'discretised'

    var_dict_anch['categorical_names'] = {var_dict_anch[vk]['order_col'] : var_dict_anch[vk]['labels'] \
                                        for vk in var_dict_anch.keys() if not var_dict_anch[vk]['class_col']}

    # create discretised versions of the training and test data
    ds_cont.X_train_matrix = disc.discretize(np.array(ds_cont.X_train))
    ds_cont.X_test_matrix = disc.discretize(np.array(ds_cont.X_test))

    # fit the Anchors explainer. Onehot encode the data. Replace the data in the ds_container
    explainer = anchtab.AnchorTabularExplainer(meta_data['class_names'], meta_data['features'], ds_cont.X_train_matrix, var_dict_anch['categorical_names'])
    explainer.train(ds_cont.X_train_matrix, ds_cont.y_train, ds_cont.X_test_matrix, ds_cont.y_test)

    ds_cont.X_train_enc = explainer.encoder.transform(ds_cont.X_train_matrix)
    ds_cont.X_test_enc = explainer.encoder.transform(ds_cont.X_test_matrix)

    ds_cont.X_train_enc_matrix = ds_cont.X_train_enc.todense()
    ds_cont.X_test_enc_matrix = ds_cont.X_test_enc.todense()

    return(ds_cont, explainer)

def Anchors_explanation(instance, explainer, forest, random_state=123, threshold=0.95):
    np.random.seed(random_state)
    exp = explainer.explain_instance(instance, forest.predict, threshold=threshold)
    return(exp)

# function to manage the whole run and evaluation
def Anchors_benchmark(forest, ds_container, meta_data,
                    anchors_explainer,
                    model,
                    n_instances=100,
                    save_path='',
                    dataset_name='',
                    precis_threshold=0.95,
                    random_state=123,
                    verbose=True):

    method = 'Anchors'
    file_stem = rt.get_file_stem(model)
    save_results_file = method + '_rnst_' + str(random_state)
    # 2. Prepare Unseen Data and Predictions
    o_print('Prepare Unseen Data and Predictions for Anchors benchmark', verbose)
    # OPTION 1 - batching (to be implemented in the new code, right now it will do just one batch)
    _, instances_matrix, instances_enc, _, labels = unseen_data_prep(ds_container,
                                            n_instances=n_instances)
    # get predictions
    preds = forest.predict(instances_enc)
    sample_labels = forest.predict(ds_container.X_train_enc) # for train estimates

    o_print('Running Anchors on each instance and collecting results', verbose)
    eval_start_time = time.asctime( time.localtime(time.time()) )
    # iterate through each instance to generate the anchors explanation
    results = [[]] * len(labels)
    evaluator = strcts.evaluator()
    for i in range(len(labels)):
        instance_id = labels.index[i]
        # if i % 10 == 0:
        o_print(str(i) + ': Working on ' + method + ' for instance ' + str(instance_id), verbose)

        # get test sample by leave-one-out on current instance
        _, loo_instances_matrix, loo_instances_enc, _, loo_true_labels = ds_container.get_loo_instances(instance_id, which_split='test')

        # get the model predicted labels
        loo_preds = forest.predict(loo_instances_enc)

        # collect the time taken
        anch_start_time = timeit.default_timer()

        # start the explanation process
        explanation = Anchors_explanation(instances_matrix[i], anchors_explainer, forest,
                                            threshold=precis_threshold,
                                            random_state=random_state)

        # the whole anchor explanation routine has run so stop the clock
        anch_end_time = timeit.default_timer()
        anch_elapsed_time = anch_end_time - anch_start_time

        # Get train and test idx (boolean) covered by the anchor
        anchor_train_idx = np.array([all_eq.all() for all_eq in ds_container.X_train_matrix[:, explanation.features()] == instances_matrix[:,explanation.features()][i]])
        anchor_test_idx = np.array([all_eq.all() for all_eq in loo_instances_matrix[:, explanation.features()] == instances_matrix[:,explanation.features()][i]])

        # create a class to run the standard evaluation
        train_metrics = evaluator.evaluate(prior_labels=sample_labels, post_idx=anchor_train_idx)
        test_metrics = evaluator.evaluate(prior_labels=loo_preds, post_idx=anchor_test_idx)

        # collect the results
        tc = [preds[i]]
        tc_lab = meta_data['get_label'](meta_data['class_col'], tc)

        true_class = ds_container.y_test.loc[instance_id]
        results[i] = [dataset_name,
            instance_id,
            method,
            ' AND '.join(explanation.names()),
            len(explanation.names()),
            true_class,
            meta_data['class_names'][true_class],
            tc[0],
            tc_lab[0],
            tc[0],
            tc_lab[0],
            0, # meaningless for boosting # np.array([tree.predict(instances_enc[i]) == tc for tree in forest.estimators_]).mean(), # majority vote share
            0, # accumulated weight not meaningful for Anchors
            test_metrics['prior']['p_counts'][tc][0],
            train_metrics['posterior'][tc][0],
            train_metrics['stability'][tc][0],
            train_metrics['recall'][tc][0],
            train_metrics['f1'][tc][0],
            train_metrics['cc'][tc][0],
            train_metrics['ci'][tc][0],
            train_metrics['ncc'][tc][0],
            train_metrics['nci'][tc][0],
            train_metrics['npv'][tc][0],
            train_metrics['accuracy'][tc][0],
            train_metrics['lift'][tc][0],
            train_metrics['coverage'],
            train_metrics['xcoverage'],
            train_metrics['kl_div'],
            test_metrics['posterior'][tc][0],
            test_metrics['stability'][tc][0],
            test_metrics['recall'][tc][0],
            test_metrics['f1'][tc][0],
            test_metrics['cc'][tc][0],
            test_metrics['ci'][tc][0],
            test_metrics['ncc'][tc][0],
            test_metrics['nci'][tc][0],
            test_metrics['npv'][tc][0],
            test_metrics['accuracy'][tc][0],
            test_metrics['lift'][tc][0],
            test_metrics['coverage'],
            test_metrics['xcoverage'],
            test_metrics['kl_div'],
            anch_elapsed_time]

        # save an intermediate file
        if save_path is not None:
            if i == 0:
                rt.save_results(cfg.results_headers, np.array(results[i]).reshape(1, -1), save_results_path=save_path,
                                save_results_file=save_results_file)
            else:
                with open(save_path + save_results_file + '.csv','a') as f:
                    writer = csv.writer(f)
                    writer.writerow([i] + results[i])

        # end for

    if save_path is not None:
        # save to file between each loop, in case of crashes/restarts
        rt.save_results(cfg.results_headers, results, save_results_path=save_path,
                        save_results_file=save_results_file)

        # collect summary_results
        with open(meta_data['get_save_path']() + file_stem + 'performance_rnst_' + str(meta_data['random_state']) + '.json', 'r') as infile:
            forest_performance = json.load(infile)
        f_perf = forest_performance[method]['test_accuracy']
        sd_f_perf = st.binom.std(len(labels), f_perf)

        summary_results = [[dataset_name, method, len(labels), 1, \
                            1, 1, 1, 0, \
                            np.mean([rl_ln[4] for rl_ln in results]), np.std([rl_ln[4] for rl_ln in results]), \
                            eval_start_time, time.asctime( time.localtime(time.time()) ), \
                            f_perf, sd_f_perf, \
                            1, 0, \
                            1, 1, 0]]

        rt.save_results(cfg.summary_results_headers, summary_results, save_path, save_results_file + '_summary')

def defragTrees_prep(forest, meta_data, ds_container,
                        Kmax=10, maxitr=100, restart=10,
                        save_path='', verbose=True):

    X_train=ds_container.X_train_enc_matrix
    y_train=ds_container.y_train
    X_test=ds_container.X_test_enc_matrix
    y_test=ds_container.y_test

    feature_names = meta_data['features_enc']
    random_state = meta_data['random_state']

    o_print('Running defragTrees', verbose)
    o_print('', verbose)
    eval_start_time = time.asctime( time.localtime(time.time()) )
    defTrees_start_time = timeit.default_timer()

    # fit simplified model
    splitter = DefragModel.parseSLtrees(forest) # parse sklearn tree ensembles into the array of (feature index, threshold)
    mdl = DefragModel(modeltype='classification', maxitr=maxitr, qitr=0, tol=1e-6, restart=restart, verbose=0, seed=random_state)
    mdl.fit(np.array(X_train), y_train, splitter, Kmax, fittype='FAB', featurename=feature_names)

    defTrees_end_time = timeit.default_timer()
    defTrees_elapsed_time = defTrees_end_time - defTrees_start_time
    o_print('Fit defragTrees time elapsed: ' + "{:0.4f}".format(defTrees_elapsed_time) + ' seconds', verbose)
    o_print('', verbose)

    score, cover, coll = mdl.evaluate(np.array(X_test), y_test)
    o_print('defragTrees test accuracy', verbose)
    o_print('Accuracy = %f' % (1 - score,), verbose)
    o_print('Coverage = %f' % (cover,), verbose)
    o_print('Overlap = %f' % (coll,), verbose)

    return(mdl, eval_start_time, defTrees_elapsed_time)

def rule_list_from_dfrgtrs(dfrgtrs):
    rule_list = [[]] * len(dfrgtrs.rule_)
    for r, rule in enumerate(dfrgtrs.rule_):
        rule_list[r] = [(dfrgtrs.featurename_[int(item[0]-1)], not item[1], item[2]) for item in rule]

    return(rule_list)

def which_rule(rule_list, X, features):
    left_idx = np.array([i for i in range(len(X.todense()))])
    rules_idx = np.full(len(left_idx), np.nan)
    rule_cascade = 0
    while len(left_idx) > 0 and rule_cascade < len(rule_list):
        rule_evaluator = strcts.rule_evaluator()
        match_idx = rule_evaluator.apply_rule(rule=rule_list[rule_cascade],
                                              instances=X[left_idx,:],
                                              features=features)
        rules_idx[left_idx[match_idx]] = rule_cascade
        left_idx = np.array([li for li, mi in zip(left_idx, match_idx) if not mi])
        rule_cascade += 1
    # what's left is the default prediction
    rules_idx[np.where(np.isnan(rules_idx))] = rule_cascade # default prediction
    rules_idx = rules_idx.astype(int)
    return(rules_idx)

def defragTrees_benchmark(forest, ds_container, meta_data, model, dfrgtrs,
                            eval_start_time, defTrees_elapsed_time,
                            n_instances=100,
                            save_path='', dataset_name='',
                            random_state=123,
                            verbose=True):

    method = 'defragTrees'
    file_stem = rt.get_file_stem(model)
    o_print('defragTrees benchmark', verbose)

    _, _, instances_enc, instances_enc_matrix, labels = unseen_data_prep(ds_container,
                                                                            n_instances=n_instances)
    forest_preds = forest.predict(instances_enc)
    dfrgtrs_preds = dfrgtrs.predict(np.array(instances_enc_matrix))

    defTrees_mean_elapsed_time = defTrees_elapsed_time / len(labels)

    eval_model = rt.evaluate_model(y_true=labels, y_pred=dfrgtrs_preds,
                        class_names=meta_data['class_names_label_order'],
                        model=model,
                        plot_cm=False, plot_cm_norm=False, # False here will output the metrics and suppress the plots
                        save_path=save_path,
                        method=method,
                        random_state=random_state)

    rule_list = rule_list_from_dfrgtrs(dfrgtrs)

    results = [[]] * len(labels)
    rule_idx = []
    evaluator = strcts.evaluator()
    for i in range(len(labels)):
        instance_id = labels.index[i]
        if i % 10 == 0: o_print(str(i) + ': Working on ' + method + ' for instance ' + str(instance_id), verbose)

        # get test sample by leave-one-out on current instance
        _, _, loo_instances_enc, loo_instances_enc_matrix, loo_true_labels = ds_container.get_loo_instances(instance_id,
                                                                                                            which_split='test')
        # get the model predicted labels
        loo_preds = forest.predict(loo_instances_enc)

        # start a timer for the individual eval
        dt_start_time = timeit.default_timer()

        # which rule appies to each loo instance
        rule_idx.append(which_rule(rule_list, loo_instances_enc, features=meta_data['features_enc']))

        # which rule appies to current instance
        rule = which_rule(rule_list, instances_enc[i], features=meta_data['features_enc'])
        if rule[0] >= len(rule_list):
            pretty_rule = []
        else:
            pretty_rule = evaluator.prettify_rule(rule_list[rule[0]], meta_data['var_dict'])

        dt_end_time = timeit.default_timer()
        dt_elapsed_time = dt_end_time - dt_start_time
        dt_elapsed_time = dt_elapsed_time + defTrees_mean_elapsed_time # add the mean modeling time per instance

        # which instances are covered by this rule
        covered_instances = rule_idx[i] == rule

        metrics = evaluator.evaluate(prior_labels=loo_preds, post_idx=covered_instances)

        # majority class is the forest vote class
        # target class is the benchmark algorithm prediction
        mc = [forest_preds[i]]
        tc = [dfrgtrs_preds[i]]
        mc_lab = meta_data['get_label'](meta_data['class_col'], mc)
        tc_lab = meta_data['get_label'](meta_data['class_col'], tc)

        true_class = ds_container.y_test.loc[instance_id]
        results[i] = [dataset_name,
        instance_id,
        method,
        pretty_rule,
        len(rule),
        true_class,
        meta_data['class_names'][true_class],
        mc[0],
        mc_lab[0],
        tc[0],
        tc_lab[0],
        0, # meaningless for boosting # np.array([tree.predict(instances_enc[i]) == mc for tree in forest.estimators_]).mean(), # majority vote share
        0, # accumulated weight not meaningful for dfrgtrs
        metrics['prior']['p_counts'][mc][0],
        metrics['posterior'][tc][0],
        metrics['stability'][tc][0],
        metrics['recall'][tc][0],
        metrics['f1'][tc][0],
        metrics['cc'][tc][0],
        metrics['ci'][tc][0],
        metrics['ncc'][tc][0],
        metrics['nci'][tc][0],
        metrics['npv'][tc][0],
        metrics['accuracy'][tc][0],
        metrics['lift'][tc][0],
        metrics['coverage'],
        metrics['xcoverage'],
        metrics['kl_div'],
        penalise_bad_prediction(mc, tc, metrics['posterior'][mc][0]),
        penalise_bad_prediction(mc, tc, metrics['stability'][mc][0]),
        penalise_bad_prediction(mc, tc, metrics['recall'][mc][0]),
        penalise_bad_prediction(mc, tc, metrics['f1'][mc][0]),
        metrics['cc'][tc][0],
        metrics['ci'][tc][0],
        metrics['ncc'][tc][0],
        metrics['nci'][tc][0],
        penalise_bad_prediction(mc, tc, metrics['npv'][mc][0]),
        penalise_bad_prediction(mc, tc, metrics['accuracy'][mc][0]),
        penalise_bad_prediction(mc, tc, metrics['lift'][mc][0]),
        metrics['coverage'],
        metrics['xcoverage'],
        metrics['kl_div'],
        dt_elapsed_time]

    if save_path is not None:
        save_results_file=method + '_rnst_' + str(random_state)

        rt.save_results(cfg.results_headers, results,
                        save_results_path=save_path,
                        save_results_file=save_results_file)

        # collect summary_results
        with open(meta_data['get_save_path']() + file_stem + 'performance_rnst_' + str(meta_data['random_state']) + '.json', 'r') as infile:
            forest_performance = json.load(infile)
        f_perf = forest_performance['main']['test_accuracy']
        sd_f_perf = st.binom.std(len(labels), f_perf)
        p_perf = np.mean(dfrgtrs_preds == labels)
        sd_p_perf = st.binom.std(len(labels), p_perf)
        fid = np.mean(dfrgtrs_preds == forest_preds)
        summary_results = [[dataset_name, method, len(labels), len(rule_list), \
                            len(np.unique(rule_idx)), np.median(np.array(rule_idx) + 1), np.mean(np.array(rule_idx) + 1), np.std(np.array(rule_idx) + 1), \
                            np.mean([rl_ln[4] for rl_ln in results]), np.std([rl_ln[4] for rl_ln in results]), \
                            eval_start_time, time.asctime( time.localtime(time.time()) ), \
                            f_perf, sd_f_perf, \
                            p_perf, sd_p_perf, \
                            eval_model['test_kappa'], fid, sqrt((fid/(1-fid))/len(labels))]]

        rt.save_results(cfg.summary_results_headers, summary_results,
                        save_results_path=save_path,
                        save_results_file=save_results_file + '_summary')

# LORE
def gpdg_record_init(x):
    return x

def gpdg_random_init(feature_values):
    individual = list()
    for feature_idx in feature_values:
        values = feature_values[feature_idx]
        val = np.random.choice(values, 1)[0]
        individual.append(val)
    return individual

def gpdg_cPickle_clone(x):
    return cPickle.loads(cPickle.dumps(x))

def gpdg_mutate(feature_values, indpb, toolbox, individual):
    new_individual = toolbox.clone(individual)
    for feature_idx in range(0, len(individual)):
        values = feature_values[feature_idx]
        if np.random.random() <= indpb:
            val = np.random.choice(values, 1)[0]
            new_individual[feature_idx] = val
    return new_individual,

def gpdg_fitness_sso(x0, x_enc, bb, alpha1, alpha2, eta, discrete, continuous, class_name, idx_features, distance_function, x1):
    # similar_same_outcome
    x0d = {idx_features[i]: val for i, val in enumerate(x0)}
    x1d = {idx_features[i]: val for i, val in enumerate(x1)}

    # zero if is too similar
    sim_ratio = 1.0 - distance_function(x0d, x1d, discrete, continuous, class_name)
    record_similarity = 0.0 if sim_ratio >= eta else sim_ratio

    y0 = bb.predict(np.asarray(x_enc.transform([x0]).todense()).reshape(1, -1))[0]
    y1 = bb.predict(np.asarray(x_enc.transform([x1]).todense()).reshape(1, -1))[0]
    target_similarity = 1.0 if y0 == y1 else 0.0

    evaluation = alpha1 * record_similarity + alpha2 * target_similarity
    return evaluation,

def gpdg_fitness_sdo(x0, x_enc, bb, alpha1, alpha2, eta, discrete, continuous, class_name, idx_features, distance_function, x1):
    # similar_different_outcome
    x0d = {idx_features[i]: val for i, val in enumerate(x0)}
    x1d = {idx_features[i]: val for i, val in enumerate(x1)}

    # zero if is too similar
    sim_ratio = 1.0 - distance_function(x0d, x1d, discrete, continuous, class_name)
    record_similarity = 0.0 if sim_ratio >= eta else sim_ratio

    y0 = bb.predict(np.asarray(x_enc.transform([x0]).todense()).reshape(1, -1))[0]
    y1 = bb.predict(np.asarray(x_enc.transform([x1]).todense()).reshape(1, -1))[0]
    target_similarity = 1.0 if y0 != y1 else 0.0

    evaluation = alpha1 * record_similarity + alpha2 * target_similarity
    return evaluation,


def gpdg_fitness_dso(x0, x_enc, bb, alpha1, alpha2, eta, discrete, continuous, class_name, idx_features, distance_function, x1):
    # dissimilar_same_outcome
    x0d = {idx_features[i]: val for i, val in enumerate(x0)}
    x1d = {idx_features[i]: val for i, val in enumerate(x1)}

    # zero if is too dissimilar
    sim_ratio = 1.0 - distance_function(x0d, x1d, discrete, continuous, class_name)
    record_similarity = 0.0 if sim_ratio <= eta else 1.0 - sim_ratio

    y0 = bb.predict(np.asarray(x_enc.transform([x0]).todense()).reshape(1, -1))[0]
    y1 = bb.predict(np.asarray(x_enc.transform([x1]).todense()).reshape(1, -1))[0]
    target_similarity = 1.0 if y0 == y1 else 0.0

    evaluation = alpha1 * record_similarity + alpha2 * target_similarity
    return evaluation,


def gpdg_fitness_ddo(x0, x_enc, bb, alpha1, alpha2, eta, discrete, continuous, class_name, idx_features, distance_function, x1):
    # dissimilar_different_outcome
    x0d = {idx_features[i]: val for i, val in enumerate(x0)}
    x1d = {idx_features[i]: val for i, val in enumerate(x1)}

    # zero if is too dissimilar
    sim_ratio = 1.0 - distance_function(x0d, x1d, discrete, continuous, class_name)
    record_similarity = 0.0 if sim_ratio <= eta else 1.0 - sim_ratio

    y0 = bb.predict(np.asarray(x_enc.transform([x0]).todense()).reshape(1, -1))[0]
    y1 = bb.predict(np.asarray(x_enc.transform([x1]).todense()).reshape(1, -1))[0]
    target_similarity = 1.0 if y0 != y1 else 0.0

    evaluation = alpha1 * record_similarity + alpha2 * target_similarity
    return evaluation,


def gpdg_setup_toolbox(record, x_enc, feature_values, bb, init, init_params, evaluate, discrete, continuous, class_name,
                  idx_features, distance_function, population_size=1000, alpha1=0.5, alpha2=0.5, eta=0.3,
                  mutpb=0.2, tournsize=3):

    creator.create("fitness", base.Fitness, weights=(1.0,))
    creator.create("individual", list, fitness=creator.fitness)

    toolbox = base.Toolbox()
    toolbox.register("feature_values", init, init_params)
    toolbox.register("individual", tools.initIterate, creator.individual, toolbox.feature_values)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=population_size)

    toolbox.register("clone", gpdg_cPickle_clone)
    toolbox.register("evaluate", evaluate, record, x_enc, bb, alpha1, alpha2, eta, discrete, continuous,
                     class_name, idx_features, distance_function)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", gpdg_mutate, feature_values, mutpb, toolbox)
    toolbox.register("select", tools.selTournament, tournsize=tournsize)

    return toolbox


def gpdg_fit(toolbox, population_size=1000, halloffame_ratio=0.1, cxpb=0.5, mutpb=0.2, ngen=10, verbose=False):
    print('fit size: ' + str(population_size))
    halloffame_size = int(np.round(population_size * halloffame_ratio))

    population = toolbox.population(n=population_size)
    halloffame = tools.HallOfFame(halloffame_size)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    population, logbook = algorithms.eaSimple(population, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen,
                                              stats=stats, halloffame=halloffame, verbose=verbose)
    print('after eaSimple')
    print(time.asctime( time.localtime(time.time()) ))

    return population, halloffame, logbook


def gpdg_get_oversample(population, halloffame):
    fitness_values = [p.fitness.wvalues[0] for p in population]
    fitness_values = sorted(fitness_values)
    fitness_diff = [fitness_values[i+1] - fitness_values[i] for i in range(0, len(fitness_values)-1)]

    index = np.max(np.argwhere(fitness_diff == np.amax(fitness_diff)).flatten().tolist())
    fitness_value_thr = fitness_values[index]
    oversample = list()

    for p in population:
        if p.fitness.wvalues[0] > fitness_value_thr:
            oversample.append(list(p))

    for h in halloffame:
        if h.fitness.wvalues[0] > fitness_value_thr:
            oversample.append(list(h))

    return oversample

def gpdg_generate_data(x, x_enc, feature_values, bb, discrete, continuous, class_name, idx_features, distance_function,
                  neigtype='all', population_size=1000, halloffame_ratio=0.1, alpha1=0.5, alpha2=0.5, eta1=1.0,
                  eta2=0.0, tournsize=3, cxpb=0.5, mutpb=0.2, ngen=10, return_logbook=False):

    if neigtype == 'all':
        neigtype = {'ss': 0.25, 'sd': 0.25, 'ds': 0.25, 'dd': 0.25}

    size_sso = int(np.round(population_size * neigtype.get('ss', 0.0)))
    size_sdo = int(np.round(population_size * neigtype.get('sd', 0.0)))
    size_dso = int(np.round(population_size * neigtype.get('ds', 0.0)))
    size_ddo = int(np.round(population_size * neigtype.get('dd', 0.0)))

    Xgp = list()

    if size_sso > 0.0:
        toolbox_sso = gpdg_setup_toolbox(x, x_enc, feature_values, bb, init=gpdg_record_init, init_params=x, evaluate=gpdg_fitness_sso,
                                    discrete=discrete, continuous=continuous, class_name=class_name,
                                    idx_features=idx_features, distance_function=distance_function,
                                    population_size=size_sso, alpha1=alpha1, alpha2=alpha2, eta=eta1, mutpb=mutpb,
                                    tournsize=tournsize)
        print('sso_fit')
        population, halloffame, logbook = gpdg_fit(toolbox_sso, population_size=size_sso, halloffame_ratio=halloffame_ratio,
                                              cxpb=cxpb, mutpb=mutpb, ngen=ngen, verbose=False)

        Xsso = gpdg_get_oversample(population, halloffame)
        if Xsso:
            Xgp.append(Xsso)

    if size_sdo > 0.0:
        toolbox_sdo = gpdg_setup_toolbox(x, x_enc, feature_values, bb, init=gpdg_record_init, init_params=x, evaluate=gpdg_fitness_sdo,
                                    discrete=discrete, continuous=continuous, class_name=class_name,
                                    idx_features=idx_features, distance_function=distance_function,
                                    population_size=size_sdo, alpha1=alpha1, alpha2=alpha2, eta=eta1, mutpb=mutpb,
                                    tournsize=tournsize)
        print('sdo_fit')
        population, halloffame, logbook = gpdg_fit(toolbox_sdo, population_size=size_sdo, halloffame_ratio=halloffame_ratio,
                                              cxpb=cxpb, mutpb=mutpb, ngen=ngen, verbose=False)

        Xsdo = gpdg_get_oversample(population, halloffame)
        if Xsdo: # can't be empty list
            Xgp.append(Xsdo)

    if size_dso > 0.0:
        toolbox_dso = gpdg_setup_toolbox(x, x_enc, feature_values, bb, init=gpdg_record_init, init_params=x, evaluate=gpdg_fitness_dso,
                                    discrete=discrete, continuous=continuous, class_name=class_name,
                                    idx_features=idx_features, distance_function=distance_function,
                                    population_size=size_dso, alpha1=alpha1, alpha2=alpha2, eta=eta2, mutpb=mutpb,
                                    tournsize=tournsize)
        print('dso_fit')
        population, halloffame, logbook = gpdg_fit(toolbox_dso, population_size=size_dso, halloffame_ratio=halloffame_ratio,
                                              cxpb=cxpb, mutpb=mutpb, ngen=ngen, verbose=False)

        Xdso = gpdg_get_oversample(population, halloffame)
        if Xdso:
            Xgp.append(Xdso)

    if size_ddo > 0.0:
        toolbox_ddo = gpdg_setup_toolbox(x, x_enc, feature_values, bb, init=gpdg_record_init, init_params=x, evaluate=gpdg_fitness_ddo,
                                    discrete=discrete, continuous=continuous, class_name=class_name,
                                    idx_features=idx_features, distance_function=distance_function,
                                    population_size=size_ddo, alpha1=alpha1, alpha2=alpha2, eta=eta2, mutpb=mutpb,
                                    tournsize=tournsize)
        print('ddo_fit')
        population, halloffame, logbook = gpdg_fit(toolbox_ddo, population_size=size_ddo, halloffame_ratio=halloffame_ratio,
                                              cxpb=cxpb, mutpb=mutpb, ngen=ngen, verbose=False)

        Xddo = gpdg_get_oversample(population, halloffame)
        if Xddo:
            Xgp.append(Xddo)

    try: # will fail if Xgp is still an empty list
        Xgp = np.concatenate((Xgp), axis=0)
    except:
        return(None)

    if return_logbook:
        return Xgp, logbook

    return Xgp

def gpdg_calculate_feature_values(X, columns, class_name, discrete, continuous, size=1000,
                             discrete_use_probabilities=False,
                             continuous_function_estimation=False):

    columns1 = list(columns)
    columns1.remove(class_name)
    feature_values = dict()
    for i, col in enumerate(columns1):
        values = X[:, i]
        if col in discrete:
            if discrete_use_probabilities:
                diff_values, counts = np.unique(values, return_counts=True)
                prob = 1.0 * counts / np.sum(counts)
                new_values = np.random.choice(diff_values, size=size, p=prob)
                new_values = np.concatenate((values, new_values), axis=0)
            else:
                diff_values = np.unique(values)
                new_values = diff_values
        elif col in continuous:
            if len(np.unique(values)) == 2 and all(np.unique(values) == np.array([0., 1.])):
                binary = True
            else:
                binary = False
            if continuous_function_estimation:
                new_values = gpdg_get_distr_values(values, size, binary)
            else:
                mu = np.mean(values)
                if binary and discrete_use_probabilities:
                    new_values = st.bernoulli.rvs(p=mu, size=size)
                elif binary and not discrete_use_probabilities:
                    diff_values = np.unique(values)
                    new_values = diff_values
                else: # suppose is gaussian
                    sigma = np.std(values)
                    new_values = np.random.normal(mu, sigma, size)
            new_values = np.concatenate((values, new_values), axis=0)
        feature_values[i] = new_values

    return feature_values


def gpdg_get_distr_values(x, size=1000, binary=False):
    if binary:
        p = np.mean(x)
        if p==0.5:
            minp = maxp = p
        else:
            pees = [p, 1-p]
            minp = pees[pees.index(min(pees))]
            maxp = pees[pees.index(max(pees))]

        distr_values = np.empty(size)
        distr_values[x == 0] = st.bernoulli.rvs(minp, size=np.sum(x == 0))
        distr_values[x == 1] = st.bernoulli.rvs(maxp, size=np.sum(x == 1))
    else:
        nbr_bins = int(np.round(gpdg_estimate_nbr_bins(x)))
        name, params = gpdg_best_fit_distribution(x, nbr_bins)
        dist = getattr(st, name)

        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]

        start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
        end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

        distr_values = np.linspace(start, end, size)

        return distr_values


# Distributions to check
DISTRIBUTIONS = [st.uniform, st.dweibull, st.exponweib, st.expon, st.exponnorm, st.gamma, st.beta, st.alpha,
                 st.chi, st.chi2, st.laplace, st.lognorm, st.norm, st.powerlaw]


def gpdg_freedman_diaconis(x):
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    n = len(x)
    h = 2.0 * iqr / n**(1.0/3.0)
    k = ceil((np.max(x) - np.min(x))/h)
    return k


def gpdg_struges(x):
    n = len(x)
    k = ceil( np.log2(n) ) + 1
    return k


def gpdg_estimate_nbr_bins(x):
    if len(x) == 1:
        return 1
    k_fd = gpdg_freedman_diaconis(x) if len(x) > 2 else 1
    k_struges = gpdg_struges(x)
    if k_fd == float('inf') or np.isnan(k_fd):
        k_fd = np.sqrt(len(x))
    k = max(k_fd, k_struges)
    return k


# Create models from data
def gpdg_best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:

        # Try to fit the distribution
        try:
                #print 'aaa'
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                except Exception:
                    pass

                # identify if this distribution is better
                # print distribution.name, sse
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

        except Exception:
            pass

    return best_distribution.name, best_params

def lore_build_df2explain(bb, X, dataset):

    columns = dataset['columns']
    features_type = dataset['features_type']
    discrete = dataset['discrete']
    label_encoder = dataset['label_encoder']
    x_enc = dataset['instance_encoder']

    y = bb.predict(np.asarray(x_enc.transform(X).todense()))
    yX = np.concatenate((y.reshape(-1, 1), X), axis=1)
    data = list()
    for i, col in enumerate(columns):
        data_col = yX[:, i]
        data_col = data_col.astype(int) if col in discrete else data_col
        data_col = data_col.astype(int) if features_type[col] == 'integer' else data_col
        data.append(data_col)
    # data = map(list, map(None, *data))
    data = [[d[i] for d in data] for i in range(0, len(data[0]))]
    dfZ = pd.DataFrame(data=data, columns=columns)
    dfZ = loreutil.label_decode(dfZ, discrete, label_encoder)
    return dfZ

def lore_get_closest_diffoutcome(df, x, x_enc, discrete, continuous, class_name, blackbox, label_encoder, distance_function,
                            k=100, diff_out_ratio=0.1):

    distances = list()
    distances_0 = list()
    idx0 = list()
    distances_1 = list()
    idx1 = list()
    Z, _ = loreutil.label_encode(df, discrete, label_encoder)
    Z = Z.iloc[:, Z.columns != class_name].values
    idx = 0
    for z, z1 in zip(df.to_dict('records'), Z):
        d = distance_function(x, z, discrete, continuous, class_name)
        distances.append(d)
        # the way this is done, it's only good for binary classification
        if blackbox.predict(x_enc.transform(z1.reshape(1, -1)).todense())[0] == 0:
            distances_0.append(d)
            idx0.append(idx)
        else:
            distances_1.append(d)
            idx1.append(idx)
        idx += 1

    idx0 = np.array(idx0)
    idx1 = np.array(idx1)

    all_indexs = np.argsort(distances).tolist()[:k]
    indexes0 = list(idx0[np.argsort(distances_0).tolist()[:k]])
    indexes1 = list(idx1[np.argsort(distances_1).tolist()[:k]])

    if 1.0 * len(set(all_indexs) & set(indexes0)) / len(all_indexs) < diff_out_ratio:
        k_index = k - int(k * diff_out_ratio)
        final_indexes = all_indexs[:k_index] + indexes0[:int(k * diff_out_ratio)]
    elif 1.0 * len(set(all_indexs) & set(indexes1)) < diff_out_ratio:
        k_index = k - int(k * diff_out_ratio)
        final_indexes = all_indexs[:k_index] + indexes1[:int(k * diff_out_ratio)]
    else:
        final_indexes = all_indexs

    return final_indexes

def lore_normalized_euclidean_distance(x, y):
    if np.var(x) + np.var(y) == 0.0:
        return 0.0
    else:
        return (0.5 * np.var(x - y) / (np.var(x) + np.var(y)))

def lore_simple_match_distance(x, y):
    count = 0
    for xi, yi in zip(x, y):
        if xi == yi:
            count += 1
    sim_ratio = 1.0 * count / len(x)

    return 1.0 - sim_ratio

def lore_mixed_distance(x, y, discrete, continuous, class_name, ddist, cdist):
    xd = [x[att] for att in discrete if att != class_name]
    wd = 0.0
    dd = 0.0
    if len(xd) > 0:
        yd = [y[att] for att in discrete if att != class_name]
        wd = 1.0 * len(discrete) / (len(discrete) + len(continuous))
        dd = ddist(xd, yd)

    xc = np.array([x[att] for att in continuous])
    wc = 0.0
    cd = 0.0
    if len(xc) > 0:
        yc = np.array([y[att] for att in continuous])
        wc = 1.0 * len(continuous) / (len(discrete) + len(continuous))
        cd = cdist(xc, yc)

    return wd * dd + wc * cd

def lore_genetic_neighborhood(dfZ, x, blackbox, dataset, popsize=1000):
    discrete = dataset['discrete']
    continuous = dataset['continuous']
    class_name = dataset['class_name']
    idx_features = dataset['idx_features']
    feature_values = dataset['feature_values']
    x_enc = dataset['instance_encoder']

    discrete_no_class = list(discrete)
    discrete_no_class.remove(class_name)

    def distance_function(x0, x1, discrete, continuous, class_name):
        return lore_mixed_distance(x0, x1, discrete, continuous, class_name,
                              ddist=lore_simple_match_distance,
                              cdist=lore_normalized_euclidean_distance)

    Z = gpdg_generate_data(x, x_enc, feature_values, blackbox, discrete_no_class, continuous, class_name, idx_features,
                      distance_function,
                      neigtype={'ss': 0.5, 'sd': 0.5},
                      population_size=popsize, halloffame_ratio=0.1,
                      alpha1=0.5, alpha2=0.5, eta1=1.0, eta2=0.0,  tournsize=3, cxpb=0.5, mutpb=0.2, ngen=10)
    if Z is None:
        return(None, Z)
    zy = blackbox.predict(x_enc.transform(Z).todense())
    if len(np.unique(zy)) == 1: # the model is predicting everything the same
        label_encoder = dataset['label_encoder']
        dfx = lore_build_df2explain(blackbox, x.reshape(1, -1), dataset).to_dict('records')[0]
        dfZ = lore_build_df2explain(blackbox, dfZ, dataset)
        neig_indexes = lore_get_closest_diffoutcome(dfZ, dfx, x_enc, discrete, continuous, class_name,
                                               blackbox, label_encoder, distance_function, k=100)
        Zn, _ = loreutil.label_encode(dfZ, discrete, label_encoder)
        Zn = Zn.iloc[neig_indexes, Zn.columns != class_name].values
        Z = np.concatenate((Z, Zn), axis=0)

    dfZ = lore_build_df2explain(blackbox, Z, dataset)
    return dfZ, Z


def lore_generate_random_data(X, class_name, columns, discrete, continuous, features_type, size=1000, uniform=True):
    if isinstance(X, pd.DataFrame):
        X = X.values
    X1 = list()
    columns1 = list(columns)
    columns1.remove(class_name)
    for i, col in enumerate(columns1):
        values = X[:, i]
        diff_values = np.unique(values)
        prob_values = [1.0 * list(values).count(val) / len(values) for val in diff_values]
        if col in discrete:
            if uniform:
                new_values = np.random.choice(diff_values, size)
            else:
                new_values = np.random.choice(diff_values, size, prob_values)
        elif col in continuous:
            mu = np.mean(values)
            sigma = np.std(values)
            if sigma <= 0.0:
                new_values = np.array([values[0]] * size)
            else:
                new_values = np.random.normal(mu, sigma, size)
        if features_type[col] == 'integer':
            new_values = new_values.astype(int)
        X1.append(new_values)
    X1 = np.concatenate((X, np.column_stack(X1)), axis=0).tolist()
    if isinstance(X, pd.DataFrame):
        X1 = pd.DataFrame(data=X1, columns=columns1)
    return X1

def lore_random_neighborhood(dfZ, x, blackbox, dataset, popsize=1000, stratified=True):
    discrete = dataset['discrete']
    continuous = dataset['continuous']
    label_encoder = dataset['label_encoder']
    class_name = dataset['class_name']
    columns = dataset['columns']
    features_type = dataset['features_type']
    x_enc = dataset['instance_encoder']

    if stratified:

        def distance_function(x0, x1, discrete, continuous, class_name):
            return lore_mixed_distance(x0, x1, discrete, continuous, class_name,
                                  ddist=lore_simple_match_distance,
                                  cdist=lore_normalized_euclidean_distance)

        dfx = lore_build_df2explain(blackbox, x.reshape(1, -1), dataset).to_dict('records')[0]
        # need to add the predictions back in
        dfZ = lore_build_df2explain(blackbox, dfZ, dataset)
        neig_indexes = lore_get_closest_diffoutcome(dfZ, dfx, x_enc, discrete, continuous, class_name,
                                               blackbox, label_encoder, distance_function, k=100)

        Z, _ = loreutil.label_encode(dfZ, discrete, label_encoder)
        Z = Z.iloc[neig_indexes, Z.columns != class_name].values
        Z = lore_generate_random_data(Z, class_name, columns, discrete, continuous, features_type, size=popsize, uniform=True)
        dfZ = lore_build_df2explain(blackbox, Z, dataset)

        return dfZ, Z

    else:
        Z, _ = lroeutil.label_encode(dfZ, discrete, label_encoder)
        Z = Z.iloc[:, Z.columns != class_name].values
        Z = lore_generate_random_data(Z, class_name, columns, discrete, continuous, features_type, size=popsize, uniform=True)
        dfZ = util.build_df2explain(blackbox, Z, dataset)

        return dfZ, Z


# modified to return a boolean index
def lore_get_covered(rule, X, dataset):
    covered_indexes = list()
    for x in X:
        covered_indexes.append(lore.is_satisfied(x, rule, dataset['discrete'], dataset['features_type']))
    return covered_indexes


def lore_explain(instance, X_train, dataset, blackbox,
                discrete_use_probabilities=False,
                continuous_function_estimation=False,
                max_popsize=1000,
                log=False, random_state=123):

    random.seed(random_state)
    class_name = dataset['class_name']
    columns = dataset['columns']
    discrete = dataset['discrete']
    continuous = dataset['continuous']
    features_type = dataset['features_type']
    label_encoder = dataset['label_encoder']
    possible_outcomes = dataset['possible_outcomes']
    path_data = dataset['path_data']
    x_enc = dataset['instance_encoder']
    X2E = X_train.to_numpy()
    x = instance.to_numpy()
    dfx = lore_build_df2explain(blackbox, x.reshape(1, -1), dataset).to_dict('records')[0] # recoded single record
    bb_outcome = blackbox.predict(np.asarray(x_enc.transform([x]).todense()))[0]

    popsize = min(X2E.shape[0], max_popsize)
    # Dataset Preprocessing
    dataset['feature_values'] = gpdg_calculate_feature_values(X2E, columns, class_name, discrete, continuous, popsize,
                                                         discrete_use_probabilities, continuous_function_estimation)

    # Generate Neighborhood - seeded by explananandum instance and training distribution
    print('starting generate genetic neighbourhood')
    print(time.asctime( time.localtime(time.time()) ))
    dfZ, Z = lore_genetic_neighborhood(X_train, x, blackbox, dataset, popsize=popsize)
    if Z is None: # failed to generate a genetic neighborhood, values insufficiently different?
        print('unable to create genetic neighbourhood')
        print('starting generate random neighbourhood')
        dfZ, Z = lore_random_neighborhood(X_train, x, blackbox, dataset, popsize=popsize)

    print('endings generate neighbourhood')
    print(time.asctime( time.localtime(time.time()) ))
    print()
    y_pred_bb = blackbox.predict(np.asarray(x_enc.transform(Z).todense()))

    # Build Decision Tree
    dt, dt_dot = loreyadt.fit(dfZ, class_name, columns, features_type, discrete, continuous,
                            filename=dataset['name'], path=path_data, sep=';', log=log)

    # Apply Decision Tree on instance to explain
    cc_outcome, rule, tree_path = loreyadt.predict_rule(dt, dfx, class_name, features_type, discrete, continuous)
    # Apply Decision Tree on neighborhood
    y_pred_cc, leaf_nodes = loreyadt.predict(dt, dfZ.to_dict('records'), class_name, features_type,
                                           discrete, continuous)

    def predict(X):
        y, ln, = loreyadt.predict(dt, X, class_name, features_type, discrete, continuous)
        return y, ln

    # Update labels if necessary
    if class_name in label_encoder:
        cc_outcome = label_encoder[class_name].transform(np.array([cc_outcome]))[0]

    if class_name in label_encoder:
        y_pred_cc = label_encoder[class_name].transform(y_pred_cc)

    # Extract Coutnerfactuals
    diff_outcome = loreutil.get_diff_outcome(bb_outcome, possible_outcomes)
    counterfactuals = loreyadt.get_counterfactuals(dt, tree_path, rule, diff_outcome,
                                                 class_name, continuous, features_type)

    explanation = (rule, counterfactuals)

    infos = {
        'bb_outcome': bb_outcome,
        'cc_outcome': cc_outcome,
        'y_pred_bb': y_pred_bb,
        'y_pred_cc': y_pred_cc,
        'dfZ': dfZ,
        'Z': Z,
        'dt': dt,
        'tree_path': tree_path,
        'leaf_nodes': leaf_nodes,
        'diff_outcome': diff_outcome,
        'predict': predict,
    }

    return(rule, counterfactuals, infos)


def lore_recognize_features_type(df, class_name):
    integer_features = list(df.select_dtypes(include=np.sctypes['int']).columns)
    double_features = list(df.select_dtypes(include=np.sctypes['float']).columns)
    string_features = list(df.select_dtypes(include=np.object).columns)
    type_features = {
        'integer': integer_features,
        'double': double_features,
        'string': string_features,
    }
    features_type = dict()
    for col in integer_features:
        features_type[col] = 'integer'
    for col in double_features:
        features_type[col] = 'double'
    for col in string_features:
        features_type[col] = 'string'

    return type_features, features_type

def lore_prepare_dataset(name, mydata, meta_data):
    # don't need tt splits
    # this object is just required to inform other routines of avalable columns and types
    df = mydata.data
    class_name = meta_data['class_col']
    # class_col must be at front for this method
    columns = [meta_data['class_col']] + meta_data['features']
    type_features, features_type = lore_recognize_features_type(df, class_name)
    dataset = {
        'name': name,
        'df': mydata.data[columns],
        'columns': columns,
        'class_name': class_name,
        'possible_outcomes': meta_data['class_names_label_order'],
        'type_features': type_features,
        'features_type': features_type,
        'discrete': [vn for vn, vt in zip(meta_data['var_names'], meta_data['var_types']) if vt == 'nominal'] ,
        'continuous': [vn for vn, vt in zip(meta_data['var_names'], meta_data['var_types']) if vt == 'continuous'] ,
        'idx_features': {i : v for i, v in enumerate(meta_data['features'])},
        'label_encoder': meta_data['le_dict'],
        'instance_encoder' : mydata.encoder,
        'path_data' : meta_data['get_save_path']()
    }

    return dataset

def lore_benchmark(forest, ds_container, meta_data, model, lore_dataset,
                            n_instances=1000,
                            save_path='', dataset_name='',
                            max_popsize=1000,
                            random_state=123,
                            verbose=True):

    method = 'lore'
    file_stem = rt.get_file_stem(model)
    save_results_file = method + '_rnst_' + str(random_state)
    o_print('lore benchmark', verbose)

    instances, _ , instances_enc, instances_enc_matrix, labels = unseen_data_prep(ds_container,
                                                                                n_instances=n_instances)
    # get predictions
    forest_preds = forest.predict(instances_enc)
    sample_labels = forest.predict(ds_container.X_train_enc) # for train estimates

    o_print('Running lore on each instance and collecting results', verbose)
    eval_start_time = time.asctime( time.localtime(time.time()) )
    # iterate through each instance to generate the anchors explanation
    results = [[]] * len(labels)
    evaluator = strcts.evaluator()

    # format for lore
    lore_X_train = lore_build_df2explain(forest, ds_container.X_train.to_numpy(), lore_dataset).to_dict('records')

    for i in range(len(labels)):
        instance_id = labels.index[i]
        #if i % 10 == 0:
        o_print(str(i) + ': Working on ' + method + ' for instance ' + str(instance_id), verbose)

        # get test sample by leave-one-out on current instance
        loo_instances, _, loo_instances_enc, _, loo_true_labels = ds_container.get_loo_instances(instance_id, which_split='test')

        # format for lore
        lore_loo_instances = lore_build_df2explain(forest, loo_instances.to_numpy(), lore_dataset).to_dict('records')

        # get the model predicted labels
        loo_preds = forest.predict(loo_instances_enc)

        # collect the time taken
        lore_start_time = timeit.default_timer()

        # start the explanation process
        rule, counterfactuals, info = lore_explain(instances.loc[instance_id],
                                    ds_container.X_train,
                                    lore_dataset,
                                    forest,
                                    max_popsize=max_popsize,
                                    log=False,
                                    random_state=random_state)

        # the whole anchor explanation routine has run so stop the clock
        lore_end_time = timeit.default_timer()
        lore_elapsed_time = lore_end_time - lore_start_time

        # Get train and test idx (boolean) covered by the rule
        lore_train_idx = lore_get_covered(rule[1], lore_X_train, lore_dataset)
        lore_test_idx = lore_get_covered(rule[1], lore_loo_instances, lore_dataset)

        # create a class to run the standard evaluation
        train_metrics = evaluator.evaluate(prior_labels=sample_labels, post_idx=lore_train_idx)
        test_metrics = evaluator.evaluate(prior_labels=loo_preds, post_idx=lore_test_idx)

        # collect the results
        tc = [forest_preds[i]]
        tc_lab = meta_data['get_label'](meta_data['class_col'], tc)

        true_class = ds_container.y_test.loc[instance_id]
        results[i] = [dataset_name,
            instance_id,
            method,
            rule[1],
            len(rule[1].keys()),
            true_class,
            meta_data['class_names'][true_class],
            tc[0],
            tc_lab[0],
            tc[0],
            tc_lab[0],
            0, # meaning for boosting # np.array([tree.predict(instances_enc[i]) == tc for tree in forest.estimators_]).mean(), # majority vote share
            0, # accumulated weight not meaningful for lore
            test_metrics['prior']['p_counts'][tc][0],
            train_metrics['posterior'][tc][0],
            train_metrics['stability'][tc][0],
            train_metrics['recall'][tc][0],
            train_metrics['f1'][tc][0],
            train_metrics['cc'][tc][0],
            train_metrics['ci'][tc][0],
            train_metrics['ncc'][tc][0],
            train_metrics['nci'][tc][0],
            train_metrics['npv'][tc][0],
            train_metrics['accuracy'][tc][0],
            train_metrics['lift'][tc][0],
            train_metrics['coverage'],
            train_metrics['xcoverage'],
            train_metrics['kl_div'],
            test_metrics['posterior'][tc][0],
            test_metrics['stability'][tc][0],
            test_metrics['recall'][tc][0],
            test_metrics['f1'][tc][0],
            test_metrics['cc'][tc][0],
            test_metrics['ci'][tc][0],
            test_metrics['ncc'][tc][0],
            test_metrics['nci'][tc][0],
            test_metrics['npv'][tc][0],
            test_metrics['accuracy'][tc][0],
            test_metrics['lift'][tc][0],
            test_metrics['coverage'],
            test_metrics['xcoverage'],
            test_metrics['kl_div'],
            lore_elapsed_time]

        # save an intermediate file
        if save_path is not None:
            if i == 0:
                rt.save_results(cfg.results_headers, np.array(results[i]).reshape(1, -1), save_results_path=save_path,
                                save_results_file=save_results_file)
            else:
                with open(save_path + save_results_file + '.csv','a') as f:
                    writer = csv.writer(f)
                    writer.writerow([i] + results[i])

    # end for

    if save_path is not None:
        # save to file between each loop, in case of crashes/restarts
        rt.save_results(cfg.results_headers, results, save_results_path=save_path,
                        save_results_file=save_results_file)

        # collect summary_results
        with open(meta_data['get_save_path']() + file_stem + 'performance_rnst_' + str(meta_data['random_state']) + '.json', 'r') as infile:
            forest_performance = json.load(infile)
        f_perf = forest_performance['main']['test_accuracy']
        sd_f_perf = st.binom.std(len(labels), f_perf)
        p_perf = f_perf # for Anchors, forest pred and Anchors target are always the same
        fid = 1 # for Anchors, forest pred and Anchors target are always the same
        summary_results = [[dataset_name, method, len(labels), 1, \
                            1, 1, 1, 0, \
                            np.mean([rl_ln[4] for rl_ln in results]), np.std([rl_ln[4] for rl_ln in results]), \
                            eval_start_time, time.asctime( time.localtime(time.time()) ), \
                            f_perf, sd_f_perf, \
                            1, 0, \
                            1, 1, 0]]

        rt.save_results(cfg.summary_results_headers, summary_results, save_path, save_results_file + '_summary')


def benchmarking_prep(datasets, model, tuning, project_dir,
                        random_state, random_state_splits,
                        do_raw=True, do_discretise=False,
                        start_instance=0, verbose=True,
                        n_cores=None):

    if n_cores is None:
        n_cores = mp.cpu_count()-4

    benchmark_items = {}
    for d_constructor in datasets:
        dataset_name = d_constructor.__name__
        o_print('Preprocessing ' + dataset_name + ' data and model for ' + d_constructor.__name__ + ' with random state = ' + str(random_state), verbose)
        # 1. Data and Forest prep
        o_print('Split data into main train-test and build forest', verbose)
        mydata = d_constructor(random_state=random_state, project_dir=project_dir)
        meta_data = mydata.get_meta()
        save_path = meta_data['get_save_path']()
        train_index, test_index = mydata.get_tt_split_idx(random_state=random_state_splits)
        tt = mydata.tt_split(train_index, test_index)

        # diagnostic for starting on a specific instance
        tt.current_row_test = start_instance
        # other methods need own copy of tt, as the internal counters need same start point
        tt_lore = deepcopy(tt)
        tt_dfrgTrs = deepcopy(tt)

        # lore specific dataset format
        lore_dataset = lore_prepare_dataset(dataset_name, mydata, meta_data)

        if do_raw:
            o_print('Train main model', verbose)
            # this will train and score the model, mathod='main' (default)
            rf = rt.forest_prep(ds_container=tt,
                        meta_data=meta_data,
                        tuning_grid=tuning['grid'],
                        model=model,
                        override_tuning=tuning['override'],
                        save_path=save_path, verbose=verbose,
                        n_cores=n_cores)
        else:
            rf = None

        if do_discretise:
            o_print('Discretise data and train model, e.g. for Anchors', verbose)
            # preprocessing - discretised continuous X matrix has been added and also needs an updated var_dict
            # plus returning the fitted explainer that holds the data distribution
            tt_anch, anchors_explainer = Anchors_preproc(ds_container=tt, meta_data=meta_data)
            # diagnostic for starting on a specific instance
            tt_anch.current_row_test = start_instance

            # override_tuning False because we want to take best params from a previous run (see try/except above)
            rf_anch = rt.forest_prep(ds_container=tt_anch,
                meta_data=meta_data,
                tuning_grid=tuning['grid'],
                model=model,
                override_tuning=False,
                save_path=save_path,
                method='Anchors', verbose=verbose,
                n_cores=n_cores)
        else:
            rf_anch = None
            tt_anch = None
            anchors_explainer = None

        # collect
        benchmark_items[dataset_name] = {'main' : {'forest' : rf, 'ds_container' : tt},
                                        'anchors' : {'forest' : rf_anch, 'ds_container' : tt_anch, 'explainer' : anchors_explainer},
                                        'dfrgTrs' : {'ds_container' : tt_dfrgTrs},
                                        'lore' : {'dataset' : lore_dataset, 'ds_container' : tt_lore},
                                        'meta_data' : meta_data}
        o_print('', verbose)
    return(benchmark_items)

def do_benchmarking(benchmark_items, verbose=True, **control):
    for b in benchmark_items:
        save_path = benchmark_items[b]['meta_data']['get_save_path']()
        try:
            save_sensitivity_path = control['save_sensitivity_path']
            save_sensitivity_path = rt.extend_path(stem=save_path, extensions=[save_sensitivity_path, \
                        'wcts_' + str(control['kwargs']['which_trees']) + \
                        '_sp_' + str(control['kwargs']['support_paths']) + \
                        '_ap_' + str(control['kwargs']['alpha_paths']) + \
                        '_dpb_' + str(control['kwargs']['disc_path_bins']) + \
                        '_dpeq_' + str(control['kwargs']['disc_path_eqcounts']) + \
                        '_sf_' + str(control['kwargs']['score_func']) + \
                        '_w_' + str(control['kwargs']['weighting']) + '_'])
        except:
            save_sensitivity_path = None
        if control['method'] == 'CHIRPS':
            CHIRPS_benchmark(forest=benchmark_items[b]['main']['forest'],
                                ds_container=benchmark_items[b]['main']['ds_container'],
                                meta_data=benchmark_items[b]['meta_data'],
                                model=control['model'],
                                n_instances=control['n_instances'],
                                forest_walk_async=control['forest_walk_async'],
                                explanation_async=control['explanation_async'],
                                save_path=save_path,
                                save_sensitivity_path=save_sensitivity_path,
                                dataset_name=b,
                                random_state=control['random_state'],
                                verbose=verbose,
                                **control['kwargs'])
        if control['method'] == 'Anchors':
            Anchors_benchmark(forest=benchmark_items[b]['anchors']['forest'],
                                ds_container=benchmark_items[b]['anchors']['ds_container'],
                                meta_data=benchmark_items[b]['meta_data'],
                                anchors_explainer=benchmark_items[b]['anchors']['explainer'],
                                model=control['model'],
                                n_instances=control['n_instances'],
                                save_path=save_path,
                                dataset_name=b,
                                random_state=control['random_state'],
                                verbose=verbose)
        if control['method'] == 'defragTrees':
            dfrgtrs, eval_start_time, defTrees_elapsed_time = defragTrees_prep(ds_container=benchmark_items[b]['dfrgTrs']['ds_container'],
                                                                    meta_data=benchmark_items[b]['meta_data'],
                                                                    forest=benchmark_items[b]['main']['forest'],
                                                                    Kmax=control['Kmax'],
                                                                    restart=control['restart'],
                                                                    maxitr=control['maxitr'],
                                                                    save_path=save_path,
                                                                    verbose=verbose)
            defragTrees_benchmark(forest=benchmark_items[b]['main']['forest'],
                                    ds_container=benchmark_items[b]['main']['ds_container'],
                                    meta_data=benchmark_items[b]['meta_data'],
                                    model=control['model'],
                                    dfrgtrs=dfrgtrs, eval_start_time=eval_start_time,
                                    defTrees_elapsed_time=defTrees_elapsed_time,
                                    n_instances=control['n_instances'],
                                    save_path=save_path, dataset_name=b,
                                    random_state=control['random_state'],
                                    verbose=verbose)

        if control['method'] == 'lore':
            lore_benchmark(forest=benchmark_items[b]['main']['forest'],
                                    ds_container=benchmark_items[b]['lore']['ds_container'],
                                    meta_data=benchmark_items[b]['meta_data'],
                                    model=control['model'],
                                    lore_dataset=benchmark_items[b]['lore']['dataset'],
                                    n_instances=control['n_instances'],
                                    save_path=save_path, dataset_name=b,
                                    max_popsize=control['max_popsize'],
                                    random_state=control['random_state'],
                                    verbose=verbose)

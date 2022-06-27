import json
import time
import timeit
import pickle
import numpy as np
import scipy.stats as st
import multiprocessing as mp
from pandas import DataFrame, Series
from imblearn.over_sampling import SMOTE
from math import sqrt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import ParameterGrid, GridSearchCV
from dask_ml.model_selection import GridSearchCV as DaskGridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, cohen_kappa_score, precision_recall_fscore_support, accuracy_score

from CHIRPS import if_nexists_make_dir, if_nexists_make_file, chisq_indep_test, p_count_corrected, o_print
from CHIRPS.plotting import plot_confusion_matrix
from CHIRPS.structures import data_preprocessor

from CHIRPS import config as cfg

# bug in sk-learn. Should be fixed in August
import warnings
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)

def get_file_stem(model='RandomForest'):
    if model=='RandomForest':
        return('rf_')
    elif model=='AdaBoost1':
        return('samme_')
    elif model=='AdaBoost2':
        return('samme.r_')
    else:
        return('gbm_')

def save_tuning_results(save_path, random_state, best_params, forest_performance, model):
    file_stem = get_file_stem(model)
    if save_path is not None:
        if_nexists_make_dir(save_path)
        with open(save_path + file_stem + 'best_params_rnst_' + str(random_state) + '.json', 'w') as outfile:
            json.dump(best_params, outfile)
        with open(save_path + file_stem + 'performance_rnst_' + str(random_state) + '.json', 'w') as outfile:
            json.dump(forest_performance, outfile)

def extend_path(stem, extensions, is_dir=False):
    # add the extension and the path separator
    for x in extensions:
        stem = stem + x + cfg.path_sep
    # just add the final extension
    if is_dir:
        return(stem)
    else:
        return(stem[:-1])

def do_rf_tuning(X, y, model, # model redundant, just here for same interface
                    grid,
                    n_cores,
                    random_state=123,
                    save_path = None):

    grid = ParameterGrid(grid)
    start_time = timeit.default_timer()

    rf = RandomForestClassifier()
    params = []

    for g in grid:
        print('Trying params: ' + str(g))
        fitting_start_time = timeit.default_timer()
        rf.set_params(n_jobs=n_cores,
            oob_score = True,
            random_state=random_state,
            **g)
        rf.fit(X, y)
        fitting_end_time = timeit.default_timer()
        fitting_elapsed_time = fitting_end_time - fitting_start_time
        oobe = rf.oob_score_
        print('Training time: ' + str(fitting_elapsed_time))
        print('Out of Bag Accuracy Score: ' + str(oobe))
        print()
        g['score'] = oobe
        params.append(g)

    elapsed = timeit.default_timer() - start_time

    params = DataFrame(params).sort_values(['score','n_estimators' ],
                                        ascending=[False, True])

    best_grid = params.loc[params['score'].idxmax()]
    best_params = {k: int(v) if k in ['n_estimators'] else v for k, v in best_grid.items()}

    forest_performance = {'score' : best_grid['score'],
                        'fitting_time' : fitting_elapsed_time}

    save_tuning_results(save_path, random_state, best_params, forest_performance, model=model)

    return(best_params, forest_performance)

def do_ada_tuning(X, y, model, grid, n_cores,
                    random_state=123,
                    save_path = None):

    start_time = timeit.default_timer()
    print('Finding best params with 10-fold CV')
    rf = DaskGridSearchCV(AdaBoostClassifier(random_state=random_state),
    param_grid=grid, n_jobs=n_cores, cv=10)
    rf.fit(X, y)
    means = rf.cv_results_['mean_test_score']
    stds = rf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, rf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    elapsed_time = timeit.default_timer() - start_time
    print('CV time: ' + str(elapsed_time))
    print()

    best_params = rf.best_params_
    best_params.update({'score' : rf.best_score_})

    forest_performance = {'score' : rf.best_score_,
                        'fitting_time' : elapsed_time}

    print(best_params)
    best_params.update({'base_estimator' : str(best_params['base_estimator'])})
    save_tuning_results(save_path, random_state, best_params, forest_performance, model=model)

    return(best_params, forest_performance)

def do_gbm_tuning(X, y, model, grid, n_cores,
                    random_state=123, save_path = None,
                    verbose=True):

    start_time = timeit.default_timer()
    o_print('Finding best params with 10-fold CV', verbose)
    rf = DaskGridSearchCV(GradientBoostingClassifier(random_state=random_state),
    param_grid=grid, n_jobs=n_cores, cv=10)
    rf.fit(X, y)
    means = rf.cv_results_['mean_test_score']
    stds = rf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, rf.cv_results_['params']):
        o_print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params), verbose)
    elapsed_time = timeit.default_timer() - start_time
    o_print('CV time: ' + str(elapsed_time), verbose)
    o_print('', verbose)

    best_params = rf.best_params_
    best_params.update({'score' : rf.best_score_})

    forest_performance = {'score' : rf.best_score_,
                        'fitting_time' : elapsed_time}

    save_tuning_results(save_path, random_state, best_params, forest_performance, model=model)

    return(best_params, forest_performance)

def tune_wrapper(X, y, model, grid,
            random_state=123,
            save_path = None,
            override_tuning=False,
            verbose=True,
            n_cores=None):

    if n_cores is None:
        n_cores = mp.cpu_count()-4

    file_stem = get_file_stem(model)
    # to do - test allowable structure of grid input
    if not override_tuning:
        try:
            with open(save_path + file_stem + 'best_params_rnst_' + str(random_state) + '.json', 'r') as infile:
                o_print('using previous tuning parameters', verbose)
                best_params = json.load(infile)
            with open(save_path + file_stem + 'performance_rnst_' + str(random_state) + '.json', 'r') as infile:
                forest_performance = json.load(infile)
            return(best_params, forest_performance)
        except:
            o_print('New grid tuning... (please wait)', verbose)

    else:
        o_print('Over-riding previous tuning parameters. New grid tuning... (please wait)', verbose)
        o_print('', verbose)

    # all of this will run in the try: existing parameters went to except
    tun_start_time = timeit.default_timer()
    if model == 'RandomForest':
        best_params, forest_performance = do_rf_tuning(X, y, model,
                                                    grid=grid,
                                                    n_cores=n_cores,
                                                    random_state=random_state,
                                                    save_path=save_path)
    elif model in ('AdaBoost1', 'AdaBoost2'):
        best_params, forest_performance = do_ada_tuning(X, y, model,
                                                    grid=grid,
                                                    n_cores=n_cores,
                                                    random_state=random_state,
                                                    save_path=save_path)
    elif model == 'GBM':
        best_params, forest_performance = do_gbm_tuning(X, y, model,
                                                    grid=grid,
                                                    n_cores=n_cores,
                                                    random_state=random_state,
                                                    save_path=save_path)
    else:
        print('not implemented')
        return()
    tun_elapsed_time = timeit.default_timer() - tun_start_time
    o_print('Tuning time elapsed: ' + "{:0.4f}".format(tun_elapsed_time) + 'seconds', verbose)
    return(best_params, forest_performance)

def update_model_performance(model, save_path, test_metrics, method, random_state):

    file_stem = get_file_stem(model)

    with open(save_path + file_stem + 'performance_rnst_' + str(random_state) + '.json', 'r') as infile:
        forest_performance = json.load(infile)
    forest_performance.update({method : test_metrics})
    with open(save_path + file_stem + 'performance_rnst_' + str(random_state) + '.json', 'w') as outfile:
        json.dump(forest_performance, outfile)

def evaluate_model(y_true, y_pred, class_names=None, model='RandomForest',
                    print_metrics=False, plot_cm=True, plot_cm_norm=True,
                    save_path=None, method = 'main', random_state=123):

    # view the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    prfs = precision_recall_fscore_support(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    coka = cohen_kappa_score(y_true, y_pred)

    test_metrics = {'confmat' : cm.tolist(),
                            'test_accuracy' : acc,
                            'test_kappa' : coka,
                            'test_prec' : prfs[0].tolist(),
                            'test_recall' : prfs[1].tolist(),
                            'test_f1' : prfs[2].tolist(),
                            'test_prior' : (prfs[3] / prfs[3].sum()).tolist(), # true labels. confmat rowsums
                            'test_posterior' : (cm.sum(axis=0) / prfs[3].sum()).tolist() } # pred labels. confmat colsums
    if print_metrics:
        print(test_metrics)

    if save_path is not None:
        update_model_performance(model, save_path, test_metrics, method, random_state)

    if plot_cm:
        plot_confusion_matrix(cm, class_names=class_names,
                              title='Confusion matrix, without normalization')
    # normalized confusion matrix
    if plot_cm_norm:
        plot_confusion_matrix(cm
                              , class_names=class_names
                              , normalize=True,
                              title='Confusion matrix normalized on rows (predicted label share)')

    return(test_metrics)

def forest_prep(ds_container, meta_data, tuning_grid,
                model='RandomForest',
                save_path=None, override_tuning=False,
                method='main',
                plot_cm=False, plot_cm_norm=False,
                verbose=True, n_cores=None):

    if n_cores is None:
        n_cores = mp.cpu_count()-4

    if meta_data['needs_balance']:
        # run SMOTE oversampling
        sm = SMOTE(random_state=12, ratio = 1.0)
        X_res, y_res = sm.fit_sample(ds_container.X_train, ds_container.y_train)
        train_res = DataFrame(X_res, columns=meta_data['features'])
        train_res[meta_data['class_col']] = Series(y_res)

        preproc = data_preprocessor()
        preproc.fit(train_res, meta_data['class_col'], meta_data['var_names'], meta_data['var_types'])
        tt_res = preproc.tt_split(test_size=1, random_state=meta_data['random_state'])

        X_train=tt_res.X_train_enc
        X_test=ds_container.X_test_enc
        y_train=tt_res.y_train
        y_test=ds_container.y_test
    else:
        X_train=ds_container.X_train_enc
        X_test=ds_container.X_test_enc
        y_train=ds_container.y_train
        y_test=ds_container.y_test

    class_names=meta_data['class_names_label_order']
    random_state=meta_data['random_state']

    best_params, forest_performance = tune_wrapper(
     X=X_train,
     y=y_train,
     model=model,
     grid=tuning_grid,
     save_path=save_path,
     override_tuning=override_tuning,
     random_state=random_state, verbose=verbose,
     n_cores=n_cores)

    if model == 'RandomForest':
        best_params.update({'oob_score' : True, 'random_state' : random_state})
        del(best_params['score'])
        rf = RandomForestClassifier()
        rf.set_params(n_jobs=n_cores, **best_params)
        rf.fit(X=X_train, y=y_train)

    elif model in ['AdaBoost1', 'AdaBoost2']:
        # convert this back from a text string
        best_params.update({'base_estimator' : eval(best_params['base_estimator'])})
        best_params.update({'random_state' : random_state})
        del(best_params['score'])
        rf = AdaBoostClassifier()
        rf.set_params(**best_params)
        rf.fit(X=X_train, y=y_train)

    elif model == 'GBM':
        # convert this back from a text string
        best_params.update({'random_state' : random_state})
        del(best_params['score'])

        rf = GradientBoostingClassifier()
        rf.set_params(**best_params)
        rf.fit(X=X_train, y=y_train)

    else:
        print('not implemented')
        return()

    o_print('Best OOB Accuracy Estimate during tuning: ' '{:0.4f}'.format(forest_performance['score']), verbose)
    o_print('Best parameters:' + str(best_params), verbose)
    o_print('', verbose)
    # the outputs of this function are:
    # cm - confusion matrix as 2d array
    # acc - accuracy of model = correctly classified instance / total number instances
    # coka - Cohen's kappa score. Accuracy adjusted for probability of correct by random guess. Useful for multiclass problems
    # prfs - precision, recall, f-score, support with the following signatures as a 2d array
    # 0 <= p, r, f <= 1. s = number of instances for each true class label (row sums of cm)
    evaluate_model(y_true=y_test, y_pred=rf.predict(X_test),
                        class_names=class_names, model=model,
                        plot_cm=plot_cm, plot_cm_norm=plot_cm_norm, # False here will output the metrics and suppress the plots
                        save_path=save_path,
                        method=method,
                        random_state=random_state)

    return(rf)

def n_instance_ceiling(ds_container, n_instances=None):
    dataset_size = len(ds_container.y_test)
    if n_instances is None:
        n_instances = dataset_size
    else:
        n_instances = min(n_instances, dataset_size)
    return(n_instances)

def save_results(headers, results, save_results_path, save_results_file):
    # create new directory if necessary
    if_nexists_make_dir(save_results_path)
    # save the tabular results to a file
    output_df = DataFrame(results, columns=headers)
    output_df.to_csv(save_results_path + save_results_file + '.csv')

def evaluate_explainers(b_CHIRPS_exp, # CHIRPS_container
                                ds_container, # data_split_container (for the test data and the LOO function
                                instance_idx, # should match the instances in the batch
                                forest,
                                meta_data,
                                dataset_name='',
                                model='RandomForest',
                                eval_start_time = time.asctime( time.localtime(time.time()) ),
                                eval_alt_labelings=False,
                                eval_rule_complements=False,
                                print_to_screen=False,
                                save_results_path=None,
                                save_results_file=None,
                                save_CHIRPS=False):

    preds = forest.predict(ds_container.X_test_enc)
    results = [[]] * len(b_CHIRPS_exp.explainers)

    for i, c in enumerate(b_CHIRPS_exp.explainers):
        print(i,c)
        # get test sample by leave-one-out on current instance
        instance_id = instance_idx[i]
        _, _, instances_enc, _, true_labels = ds_container.get_loo_instances(instance_id)
        # get the model predicted labels
        labels = Series(forest.predict(instances_enc), index = true_labels.index)

        # get the detail of the current index
        _, _, current_instance_enc, _, current_instance_label = ds_container.get_by_id([instance_id], which_split='test')

        # then evaluating rule metrics on the leave-one-out test set
        eval_rule = c.evaluate_rule(rule='pruned', sample_instances=instances_enc, sample_labels=labels)
        tc = c.target_class
        tc_lab = c.target_class_label

        # collect results
        tt_prior = (labels.value_counts() / len(labels)).values
        tt_prior_counts = eval_rule['prior']['counts']
        tt_posterior = eval_rule['posterior']
        tt_posterior_counts = eval_rule['counts']
        tt_chisq = chisq_indep_test(tt_posterior_counts, tt_prior_counts)[1]
        tt_prec = eval_rule['posterior'][tc]
        tt_stab = eval_rule['stability'][tc]
        tt_recall = eval_rule['recall'][tc]
        tt_f1 = eval_rule['f1'][tc]
        tt_cc = eval_rule['cc'][tc]
        tt_ci = eval_rule['ci'][tc]
        tt_ncc = eval_rule['ncc'][tc]
        tt_nci = eval_rule['nci'][tc]
        tt_npv = eval_rule['npv'][tc]
        tt_acc = eval_rule['accuracy'][tc]
        tt_lift = eval_rule['lift'][tc]
        tt_coverage = eval_rule['coverage']
        tt_xcoverage = eval_rule['xcoverage']
        tt_kl_div = eval_rule['kl_div']

        # the rule complements to be assessed on the train set: it's out put for the user.
        if eval_rule_complements:
            rule_complement_results = c.eval_rule_complements(sample_instances=ds_container.X_train_enc, sample_labels=ds_container.y_train)

        if eval_alt_labelings:
            # get the current instance being explained
            # get_by_id takes a list of instance ids. Here we have just a single integer
            alt_labelings_results = c.get_alt_labelings(instance=current_instance_enc,
                                                        sample_instances=instances_enc,
                                                        forest=forest)
        true_class = ds_container.y_test.loc[instance_id]
        print(c.target_class_label)
        results[i] = [dataset_name,
            instance_id,
            c.algorithm,
            c.pretty_rule,
            c.rule_len,
            true_class,
            meta_data['class_names'][true_class],
            preds[i],
            meta_data['class_names'][preds[i]],
            c.target_class,
            c.target_class_label,#IOANNIS removed the [0]
            c.forest_vote_share,
            c.accumulated_weights,
            c.prior[tc],
            c.est_prec,
            c.est_stab,
            c.est_recall,
            c.est_f1,
            c.est_cc,
            c.est_ci,
            c.est_ncc,
            c.est_nci,
            c.est_npv,
            c.est_acc,
            c.est_lift,
            c.est_coverage,
            c.est_xcoverage,
            c.est_kl_div,
            tt_prec,
            tt_stab,
            tt_recall,
            tt_f1,
            tt_cc,
            tt_ci,
            tt_ncc,
            tt_nci,
            tt_npv,
            tt_acc,
            tt_lift,
            tt_coverage,
            tt_xcoverage,
            tt_kl_div,
            c.elapsed_time]

        if print_to_screen:
            print('INSTANCE RESULTS')
            print('instance id: ' + str(instance_id) + ' with true class label: ' + str(current_instance_label.values[0]) + \
                    ' (' + c.get_label(c.class_col, current_instance_label.values) + ')')
            print()
            c.to_screen()
            print('Results - Previously Unseen Sample')
            print('target class prior (unseen data): ' + str(tt_prior[tc]))
            print('rule coverage (unseen data): ' + str(tt_coverage))
            print('rule xcoverage (unseen data): ' + str(tt_xcoverage))
            print('rule precision (unseen data): ' + str(tt_prec))
            print('rule stability (unseen data): ' + str(tt_stab))
            print('rule recall (unseen data): ' + str(tt_recall))
            print('rule f1 score (unseen data): ' + str(tt_f1))
            print('rule NPV (unseen data): ' + str(tt_npv))
            print('rule lift (unseen data): ' + str(tt_lift))
            print('prior (unseen data): ' + str(tt_prior))
            print('prior counts (unseen data): ' + str(tt_prior_counts))
            print('rule posterior (unseen data): ' + str(tt_posterior))
            print('rule posterior counts (unseen data): ' + str(tt_posterior_counts))
            print('rule chisq p-value (unseen data): ' + str(tt_chisq))
            print('rule Kullback-Leibler divergence (unseen data): ' + str(tt_kl_div))
            print('Evaluation Time: ' + str(c.elapsed_time))
            print()
            if eval_rule_complements:
                print('COUNTER FACTUAL RESULTS')
                for rcr in rule_complement_results:
                    eval_rule = rcr['eval']
                    tt_prior = eval_rule['prior']['p_counts']
                    tt_prior_counts = eval_rule['prior']['counts']
                    tt_posterior = eval_rule['posterior']
                    tt_posterior_counts = eval_rule['counts']
                    tt_chisq = chisq_indep_test(tt_posterior_counts, tt_prior_counts)[1]
                    tt_prec = eval_rule['posterior'][tc]
                    tt_stab = eval_rule['stability'][tc]
                    tt_recall = eval_rule['recall'][tc]
                    tt_f1 = eval_rule['f1'][tc]
                    tt_npv = eval_rule['npv'][tc]
                    tt_acc = eval_rule['accuracy'][tc]
                    tt_lift = eval_rule['lift'][tc]
                    tt_coverage = eval_rule['coverage']
                    tt_xcoverage = eval_rule['xcoverage']
                    kl_div = rcr['kl_div']
                    print('Feature Reversed: ' + rcr['feature'])
                    print('rule: ' + rcr['pretty_rule'])
                    print('rule coverage (training data): ' + str(tt_coverage))
                    print('rule xcoverage (training data): ' + str(tt_xcoverage))
                    print('rule precision (training data): ' + str(tt_prec))
                    print('rule stability (training data): ' + str(tt_stab))
                    print('rule recall (training data): ' + str(tt_recall))
                    print('rule f1 score (training data): ' + str(tt_f1))
                    print('rule NPV (training data): ' + str(tt_npv))
                    print('rule lift (training data): ' + str(tt_lift))
                    print('prior (training data): ' + str(tt_prior))
                    print('prior counts (training data): ' + str(tt_prior_counts))
                    print('rule posterior (training data): ' + str(tt_posterior))
                    print('rule posterior counts (training data): ' + str(tt_posterior_counts))
                    print('rule chisq p-value (training data): ' + str(tt_chisq))
                    print('rule Kullback-Leibler divergence from original: ' + str(kl_div))
                    if eval_alt_labelings:
                        for alt_labels in alt_labelings_results:
                            if alt_labels['feature'] == rcr['feature']:
                                print('predictions for this rule complement')
                                if not alt_labels['mask_cover']:
                                    print('note: this combination does not exist in the unseen data sample \
                                    \nexercise caution when interpreting the results.')
                                print('instance specific. expected class: ' + str(np.argmax(alt_labels['is_mask']['p_counts'])) + \
                                        ' (' + c.get_label(c.class_col, [np.argmax(alt_labels['is_mask']['p_counts'])]) + ')')
                                print('classes: ' + str(alt_labels['is_mask']['labels']))
                                print('counts: ' + str(alt_labels['is_mask']['counts']))
                                print('proba: ' + str(alt_labels['is_mask']['p_counts']))
                                print('allowed values. expected class: ' + str(np.argmax(alt_labels['av_mask']['p_counts'])) + \
                                        ' (' + c.get_label(c.class_col, [np.argmax(alt_labels['av_mask']['p_counts'])]) + ')')
                                print('classes: ' + str(alt_labels['av_mask']['labels']))
                                print('counts: ' + str(alt_labels['av_mask']['counts']))
                                print('proba: ' + str(alt_labels['av_mask']['p_counts']))
                                print()
                    else:
                        print()

        # end for

    if save_results_path is not None:
        # save to file between each loop, in case of crashes/restarts
        save_results(cfg.results_headers, results, save_results_path, save_results_file)

        # collect summary_results
        file_stem = get_file_stem(model)
        with open(meta_data['get_save_path']() + file_stem + 'performance_rnst_' + str(meta_data['random_state']) + '.json', 'r') as infile:
            forest_performance = json.load(infile)
        f_perf = forest_performance['main']['test_accuracy']
        p_perf = f_perf # for CHIRPS, forest pred and CHIRPS target are always the same
        fid = 1 # for CHIRPS, forest pred and CHIRPS target are always the same
        sd_f_perf = st.binom.std(len(b_CHIRPS_exp.explainers), f_perf)

        # save summary results
        summary_results = [[dataset_name, results[0][2], len(b_CHIRPS_exp.explainers), 1, \
                            1, 1, 1, 0, \
                            np.mean([rl_ln[4] for rl_ln in results]), np.std([rl_ln[4] for rl_ln in results]), \
                            eval_start_time, time.asctime( time.localtime(time.time()) ), \
                            f_perf, sd_f_perf, \
                            1, 0, \
                            1, 1, 0]]

        save_results(cfg.summary_results_headers, summary_results, save_results_path, save_results_file + '_summary')

    if save_CHIRPS:
        # save the CHIRPS_container object
        explainers_store = open(save_results_path + save_results_file + '.pickle', "wb")
        pickle.dump(b_CHIRPS_exp.explainers, explainers_store)
        explainers_store.close()

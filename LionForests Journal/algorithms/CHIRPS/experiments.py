import json
import time
import timeit
import pickle
import numpy as np
from pandas import DataFrame, Series
import multiprocessing as mp
from CHIRPS import p_count, p_count_corrected
import CHIRPS.datasets as ds
from CHIRPS.structures import forest_walker
from CHIRPS.routines import tune_rf, train_rf, evaluate_model, CHIRPS_explanation, anchors_preproc, anchors_explanation
from scipy.stats import chi2_contingency
from math import sqrt
from sklearn.metrics import confusion_matrix, cohen_kappa_score, precision_recall_fscore_support, accuracy_score


from anchor import anchor_tabular as anchtab
from lime import lime_tabular as limtab


# bug in sk-learn. Should be fixed in August
import warnings
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)

def experiment(grid_idx, dataset, random_state, add_trees,
                    override_tuning, n_instances, n_batches,
                    eval_model, alpha_scores, alpha_paths,
                    support_paths, precis_threshold, run_anchors,
                    which_trees, disc_path_bins, disc_path_eqcounts,
                    iv_low, iv_high,
                    weighting, greedy,
                    forest_walk_async, chirps_explanation_async,
                    project_dir, save_rule_accs):

    print('Starting new run for ' + str(dataset) + ':')
    print('random_state ' + str(random_state) + ' and ' + str(add_trees) + ' additional trees')
    # load data set
    mydata = dataset(random_state=random_state, project_dir=project_dir)

    # train test split - one off hard-coded
    # the random_state is not used here. Just use to develop the forests
    tt = mydata.tt_split(random_state=123)

    ################ PARAMETER TUNING ###################
    ############ Only runs when required ################
    #####################################################

    best_params = tune_rf(tt['X_train_enc'], tt['y_train'],
     save_path = mydata.make_save_path(),
     random_state=mydata.random_state,
     override_tuning=override_tuning)

    #####################################################

    # update best params according to expermental design
    best_params['n_estimators'] = best_params['n_estimators'] + add_trees

    # train a rf model
    rf, enc_rf = train_rf(X=tt['X_train_enc'], y=tt['y_train'],
     best_params=best_params,
     encoder=tt['encoder'],
     random_state=mydata.random_state)

    if eval_model:
        cm, acc, coka, prfs = evaluate_model(prediction_model=enc_rf, X=tt['X_test'], y=tt['y_test'],
                     class_names=mydata.get_label(mydata.class_col, [i for i in range(len(mydata.class_names))]).tolist(),
                     plot_cm=True, plot_cm_norm=True)
    else:
        cm, acc, coka, prfs = evaluate_model(prediction_model=enc_rf, X=tt['X_test'], y=tt['y_test'],
                     class_names=mydata.get_label(mydata.class_col, [i for i in range(len(mydata.class_names))]).tolist(),
                     plot_cm=False, plot_cm_norm=False)

    # fit the forest_walker
    f_walker = forest_walker(forest = rf,
     data_container=mydata,
     encoder=tt['encoder'],
     prediction_model=enc_rf)

    # run the batch based forest walker
    getter = batch_getter(instances=tt['X_test'], labels=tt['y_test'])

    # faster to do one batch, avoids the overhead of setting up many but consumes more mem
    # get a round number of instances no more than what's available in the test set
    n_instances = min(n_instances, len(tt['y_test']))
    batch_size = int(n_instances / n_batches)
    n_instances = batch_size * n_batches

    wb_start_time = timeit.default_timer()

    # collect completed rule_acc_lite objects for the whole batch
    explainers, result_sets = CHIRPS_explanation(f_walker=f_walker,
     getter=getter,
     data_container=mydata,
     encoder=tt['encoder'],
     sample_instances=tt['X_train'],
     sample_labels=tt['y_train'],
     support_paths=support_paths,
     alpha_paths=alpha_paths,
     disc_path_bins=disc_path_bins,
     disc_path_eqcounts=disc_path_eqcounts,
     alpha_scores=alpha_scores,
     which_trees=which_trees,
     precis_threshold=precis_threshold,
     batch_size=batch_size,
     n_batches=n_batches,
     weighting=weighting,
     greedy=greedy,
     forest_walk_async=forest_walk_async,
     chirps_explanation_async=chirps_explanation_async)

    wb_end_time = timeit.default_timer()
    wb_elapsed_time = wb_end_time - wb_start_time

    wbres_start_time = timeit.default_timer()

    headers = ['instance_id', 'result_set',
                'pretty rule', 'rule length',
                'pred class', 'pred class label',
                'target class', 'target class label',
                'majority voting trees', 'majority vote share', 'pred prior',
                'precision(tr)', 'recall(tr)', 'f1(tr)',
                'accuracy(tr)', 'lift(tr)',
                'total coverage(tr)',
                'precision(tt)', 'recall(tt)', 'f1(tt)',
                'accuracy(tt)', 'lift(tt)',
                'total coverage(tt)', 'model_acc', 'model_ck']
    output = [[]] * len(explainers) * len(result_sets)

    # get all the label predictions done
    predictor = rule_tester(data_container=mydata,
                            rule=[], # dummy rule, not relevant
                            sample_instances=tt['X_test'])
    pred_instances, pred_labels = predictor.encode_pred(prediction_model=enc_rf, bootstrap=False)
    # leave one out encoder for test set evaluation
    looe = loo_encoder(pred_instances, pred_labels, tt['encoder'])

    # iterate over all the test instances to determine the various scores using leave-one-out testing
    print('evaluating found explanations')
    for i in range(len(explainers)):
        # these are the same for a whole result set
        instance_id = explainers[i][0].instance_id
        mc = explainers[i][0].major_class
        mc_lab = explainers[i][0].major_class_label
        tc = explainers[i][0].target_class
        tc_lab = explainers[i][0].target_class_label
        vt = explainers[i][0].model_votes['counts'][tc]
        mvs = explainers[i][0].model_post[tc]
        prior = explainers[i][0].posterior[0][tc]
        for j, rs in enumerate(result_sets):
            rule = explainers[i][j].pruned_rule
            prettify_rule = mydata.prettify_rule(rule)
            rule_len = len(rule)
            tr_prec = list(reversed(explainers[i][j].posterior))[0][tc]
            tr_recall = list(reversed(explainers[i][j].recall))[0][tc]
            tr_f1 = list(reversed(explainers[i][j].f1))[0][tc]
            tr_acc = list(reversed(explainers[i][j].accuracy))[0][tc]
            tr_lift = list(reversed(explainers[i][j].lift))[0][tc]
            tr_coverage = list(reversed(explainers[i][j].coverage))[0]

            # get test sample ready by leave-one-out then evaluating
            instances, enc_instances, labels = looe.loo_encode(instance_id)
            rt = rule_tester(data_container=mydata, rule=rule,
                                sample_instances=enc_instances,
                                sample_labels=labels)
            eval_rule = rt.evaluate_rule()

            # collect results
            tt_prec = eval_rule['post'][tc]
            tt_recall = eval_rule['recall'][tc]
            tt_f1 = eval_rule['f1'][tc]
            tt_acc = eval_rule['accuracy'][tc]
            tt_lift = eval_rule['lift'][tc]
            tt_coverage = eval_rule['coverage']

            output[j * len(explainers) + i] = [instance_id,
                    rs,
                    prettify_rule,
                    rule_len,
                    mc,
                    mc_lab,
                    tc,
                    tc_lab,
                    vt,
                    mvs,
                    prior,
                    tr_prec,
                    tr_recall,
                    tr_f1,
                    tr_acc,
                    tr_lift,
                    tr_coverage,
                    tt_prec,
                    tt_recall,
                    tt_f1,
                    tt_acc,
                    tt_lift,
                    tt_coverage,
                    acc,
                    coka]

    wbres_end_time = timeit.default_timer()
    wbres_elapsed_time = wbres_end_time - wbres_start_time
    print('CHIRPS batch results eval time elapsed:', "{:0.4f}".format(wbres_elapsed_time), 'seconds')
    # this completes the CHIRPS runs

    # run anchors if requested
    anch_elapsed_time = None # optional no timing
    if run_anchors:
        print('running anchors for random_state ' + str(mydata.random_state))
        # collect timings
        anch_start_time = timeit.default_timer()
        instance_ids = tt['X_test'].index.tolist() # record of row indices will be lost after preproc
        mydata, tt, explanation = anchors_preproc(dataset, random_state, iv_low, iv_high)

        rf, enc_rf = train_rf(tt['X_train_enc'], y=tt['y_train'],
        best_params=best_params,
        encoder=tt['encoder'],
        random_state=mydata.random_state)

        # collect model prediction performance stats
        if eval_model:
            cm, acc, coka, prfs = evaluate_model(prediction_model=enc_rf, X=tt['X_test'], y=tt['y_test'],
                         class_names=mydata.class_names,
                         plot_cm=True, plot_cm_norm=True)
        else:
            cm, acc, coka, prfs = evaluate_model(prediction_model=enc_rf, X=tt['X_test'], y=tt['y_test'],
                         class_names=mydata.class_names,
                         plot_cm=False, plot_cm_norm=False)

        # iterate through each instance to generate the anchors explanation
        output_anch = [[]] * n_instances
        for i in range(n_instances):
            instance_id = instance_ids[i]
            if i % 10 == 0: print('Working on Anchors for instance ' + str(instance_id))
            instance = tt['X_test'][i]
            exp = anchors_explanation(instance, explanation, rf, threshold=precis_threshold)
            # capture the explainer
            print(exp)
            explainers[i].append(exp)

            # Get test examples where the anchor applies
            fit_anchor_train = np.where(np.all(tt['X_train'][:, exp.features()] == instance[exp.features()], axis=1))[0]
            fit_anchor_test = np.where(np.all(tt['X_test'][:, exp.features()] == instance[exp.features()], axis=1))[0]
            fit_anchor_test = [fat for fat in fit_anchor_test if fat != i] # exclude current instance

            # train
            priors = p_count_corrected(tt['y_train'], [i for i in range(len(mydata.class_names))])
            if any(fit_anchor_train):
                p_counts = p_count_corrected(enc_rf.predict(tt['X_train'][fit_anchor_train]), [i for i in range(len(mydata.class_names))])
            else:
                p_counts = p_count_corrected([None], [i for i in range(len(mydata.class_names))])
            counts = p_counts['counts']
            labels = p_counts['labels']
            post = p_counts['p_counts']
            p_corrected = np.array([p if p > 0.0 else 1.0 for p in post])
            cover = counts.sum() / priors['counts'].sum()
            recall = counts/priors['counts'] # recall
            r_corrected = np.array([r if r > 0.0 else 1.0 for r in recall]) # to avoid div by zeros
            observed = np.array((counts, priors['counts']))
            if counts.sum() > 0: # previous_counts.sum() == 0 is impossible
                chisq = chi2_contingency(observed=observed[:, np.where(observed.sum(axis=0) != 0)], correction=True)
            else:
                chisq = np.nan
            f1 = [2] * ((post * recall) / (p_corrected + r_corrected))
            not_covered_counts = counts + (np.sum(priors['counts']) - priors['counts']) - (np.sum(counts) - counts)
            accu = not_covered_counts/priors['counts'].sum()
            # to avoid div by zeros
            pri_corrected = np.array([pri if pri > 0.0 else 1.0 for pri in priors['p_counts']])
            pos_corrected = np.array([pos if pri > 0.0 else 0.0 for pri, pos in zip(priors['p_counts'], post)])
            if counts.sum() == 0:
                rec_corrected = np.array([0.0] * len(pos_corrected))
                cov_corrected = np.array([1.0] * len(pos_corrected))
            else:
                rec_corrected = counts / counts.sum()
                cov_corrected = np.array([counts.sum() / priors['counts'].sum()])

            lift = pos_corrected / ( ( cov_corrected ) * pri_corrected )

            # capture train
            mc = enc_rf.predict(tt['X_test'][i].reshape(1, -1))[0]
            mc_lab = mydata.class_names[enc_rf.predict(tt['X_test'][i].reshape(1, -1))[0]]
            tc = enc_rf.predict(tt['X_test'][i].reshape(1, -1))[0]
            tc_lab = mydata.class_names[enc_rf.predict(tt['X_test'][i].reshape(1, -1))[0]]
            vt = np.nan
            mvs = np.nan
            prior = priors['p_counts'][tc]
            prettify_rule = ' AND '.join(exp.names())
            rule_len = len(exp.names())
            tr_prec = post[tc]
            tr_recall = recall[tc]
            tr_f1 = f1[tc]
            tr_acc = accu[tc]
            tr_lift = lift[tc]
            tr_coverage = cover

            # test
            priors = p_count_corrected(tt['y_test'], [i for i in range(len(mydata.class_names))])
            if any(fit_anchor_test):
                p_counts = p_count_corrected(enc_rf.predict(tt['X_test'][fit_anchor_test]), [i for i in range(len(mydata.class_names))])
            else:
                p_counts = p_count_corrected([None], [i for i in range(len(mydata.class_names))])
            counts = p_counts['counts']
            labels = p_counts['labels']
            post = p_counts['p_counts']
            p_corrected = np.array([p if p > 0.0 else 1.0 for p in post])
            cover = counts.sum() / priors['counts'].sum()
            recall = counts/priors['counts'] # recall
            r_corrected = np.array([r if r > 0.0 else 1.0 for r in recall]) # to avoid div by zeros
            observed = np.array((counts, priors['counts']))
            if counts.sum() > 0: # previous_counts.sum() == 0 is impossible
                chisq = chi2_contingency(observed=observed[:, np.where(observed.sum(axis=0) != 0)], correction=True)
            else:
                chisq = np.nan
            f1 = [2] * ((post * recall) / (p_corrected + r_corrected))
            not_covered_counts = counts + (np.sum(priors['counts']) - priors['counts']) - (np.sum(counts) - counts)
            # accuracy = (TP + TN) / num_instances formula: https://books.google.co.uk/books?id=ubzZDQAAQBAJ&pg=PR75&lpg=PR75&dq=rule+precision+and+coverage&source=bl&ots=Aa4Gj7fh5g&sig=6OsF3y4Kyk9KlN08OPQfkZCuZOc&hl=en&sa=X&ved=0ahUKEwjM06aW2brZAhWCIsAKHY5sA4kQ6AEIUjAE#v=onepage&q=rule%20precision%20and%20coverage&f=false
            accu = not_covered_counts/priors['counts'].sum()
            pri_corrected = np.array([pri if pri > 0.0 else 1.0 for pri in priors['p_counts']]) # to avoid div by zeros
            pos_corrected = np.array([pos if pri > 0.0 else 0.0 for pri, pos in zip(priors['p_counts'], post)]) # to avoid div by zeros
            if counts.sum() == 0:
                rec_corrected = np.array([0.0] * len(pos_corrected))
                cov_corrected = np.array([1.0] * len(pos_corrected))
            else:
                rec_corrected = counts / counts.sum()
                cov_corrected = np.array([counts.sum() / priors['counts'].sum()])

            lift = pos_corrected / ( ( cov_corrected ) * pri_corrected )

            # capture test
            tt_prec = post[tc]
            tt_recall = recall[tc]
            tt_f1 = f1[tc]
            tt_acc = accu[tc]
            tt_lift = lift[tc]
            tt_coverage = cover

            output_anch[i] = [instance_id,
                                'anchors', # result_set
                                prettify_rule,
                                rule_len,
                                mc,
                                mc_lab,
                                tc,
                                tc_lab,
                                mvs,
                                prior,
                                tr_prec,
                                tr_recall,
                                tr_f1,
                                tr_acc,
                                tr_lift,
                                tr_coverage,
                                tt_prec,
                                tt_recall,
                                tt_f1,
                                tt_acc,
                                tt_lift,
                                tt_coverage,
                                acc,
                                coka]

        output = np.concatenate((output, output_anch), axis=0)
        anch_end_time = timeit.default_timer()
        anch_elapsed_time = anch_end_time - anch_start_time

    # save the tabular results to a file
    output_df = DataFrame(output, columns=headers)
    output_df.to_csv(mydata.make_save_path(mydata.pickle_dir.replace('pickles', 'results') + '_rnst_' + str(mydata.random_state) + "_addt_" + str(add_trees) + '_timetest.csv'))
    # save the full rule_acc_lite objects
    if save_rule_accs:
        explainers_store = open(mydata.make_save_path('explainers' + '_rnst_' + str(mydata.random_state) + "_addt_" + str(add_trees) + '.pickle'), "wb")
        pickle.dump(explainers, explainers_store)
        explainers_store.close()

    print('Completed experiment for ' + str(dataset) + ':')
    print('random_state ' + str(mydata.random_state) + ' and ' +str(add_trees) + ' additional trees')
    # pass the elapsed times up to the caller
    return(wb_elapsed_time + wbres_elapsed_time, anch_elapsed_time, grid_idx)

def grid_experiment_mp(grid):
    # capture timing results
    start_time = timeit.default_timer()

    print(str(len(grid.index)) + ' runs to do')
    # iterate over the paramaters for each run
    for g in range(len(grid.index)):
        run_start_time = timeit.default_timer()
        experiment(grid_idx = grid.loc[g].grid_idx,
                dataset = grid.loc[g].dataset,
                random_state = grid.loc[g].random_state,
                add_trees = grid.loc[g].add_trees,
                override_tuning = grid.loc[g].override_tuning,
                n_instances = grid.loc[g].n_instances,
                n_batches = grid.loc[g].n_batches,
                eval_model = grid.loc[g].eval_model,
                alpha_scores = grid.loc[g].alpha_scores,
                alpha_paths = grid.loc[g].alpha_paths,
                support_paths = grid.loc[g].support_paths,
                precis_threshold = grid.loc[g].precis_threshold,
                run_anchors = grid.loc[g].run_anchors,
                which_trees = grid.loc[g].which_trees,
                disc_path_bins = grid.loc[g].disc_path_bins,
                disc_path_eqcounts = grid.loc[g].disc_path_eqcounts,
                iv_low = grid.loc[g].iv_low,
                iv_high = grid.loc[g].iv_high,
                weighting = grid.loc[g].weighting,
                greedy = grid.loc[g].greedy,
                forest_walk_async = grid.loc[g].forest_walk_async,
                chirps_explanation_async = grid.loc[g].chirps_explanation_async,
                project_dir = grid.loc[g].project_dir,
                save_rule_accs = grid.loc[g].save_rule_accs)
        run_elapsed_time = timeit.default_timer() - run_start_time
        cum_elapsed_time = timeit.default_timer() - start_time
        print('Run time elapsed:', "{:0.4f}".format(run_elapsed_time), 'seconds')
        print('Cumulative time elapsed:', "{:0.4f}".format(cum_elapsed_time), 'seconds')
        print()
        print()
    print('Completed ' + str(len(grid)) + ' run(s)')
    print()
    # capture timing results
    elapsed = timeit.default_timer() - start_time
    print('Time elapsed:', "{:0.4f}".format(elapsed), 'seconds')

    return('completed')

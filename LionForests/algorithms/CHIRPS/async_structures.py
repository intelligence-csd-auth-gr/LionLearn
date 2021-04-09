# this module is required for parallel processing
# parallel requires functions/classes to be in __main__ or already referenced.
# as this code is quite complex, the latter is preferred
import time
import timeit
import math
import numpy as np
from scipy import sparse
from copy import deepcopy
from scipy.stats import chi2_contingency

# parallelisable function for the forest_walker class
def async_classification_tree_walk(tree_idx, instances, labels, n_instances,
                tree_pred, tree_pred_labels,
                tree_pred_proba,
                tree_agree_maj_vote,
                feature, threshold, path, features, est_wt):

    # object for the results
    tree_paths = [{}] * n_instances

    # case that tree is a single node stump (rare in RF, normal in AdaBoost)
    if len(feature) == 1:
        for ic in range(n_instances):
            if labels is None:
                pred_class = None
            else:
                pred_class = labels[ic]
            tree_paths[ic] = {'estimator_weight' : est_wt,
                                    'pred_class' : tree_pred[ic].astype(np.int64),
                                    'pred_class_label' : tree_pred_labels[ic],
                                    'pred_proba' : tree_pred_proba[ic].tolist(),
                                    'forest_pred_class' : pred_class,
                                    'agree_maj_vote' : tree_agree_maj_vote[ic],
                                    'path' : {'feature_idx' : [],
                                                            'feature_name' : [],
                                                            'feature_value' : [],
                                                            'threshold' : [],
                                                            'leq_threshold' : []
                                                }
                                    }
    # usual case
    else:
        ic = -1 # instance_count
        for p in path:
            if feature[p] < 0: # leaf node
                continue
            if features is None:
                feature_name = None
            else:
                feature_name = features[feature[p]]
            if p == 0: # root node
                ic += 1
                feature_value = instances[ic, [feature[p]]].item(0)
                leq_threshold = feature_value <= threshold[p]
                if labels is None:
                    pred_class = None
                else:
                    pred_class = labels[ic]
                tree_paths[ic] = {'estimator_weight' : est_wt,
                                        'pred_class' : tree_pred[ic].astype(np.int64),
                                        'pred_class_label' : tree_pred_labels[ic],
                                        'pred_proba' : tree_pred_proba[ic].tolist(),
                                        'forest_pred_class' : pred_class,
                                        'agree_maj_vote' : tree_agree_maj_vote[ic],
                                        'path' : {'feature_idx' : [feature[p]],
                                                                'feature_name' : [feature_name],
                                                                'feature_value' : [feature_value],
                                                                'threshold' : [threshold[p]],
                                                                'leq_threshold' : [leq_threshold]
                                                    }
                                        }
            else:
                feature_value = instances[ic, [feature[p]]].item(0)
                leq_threshold = feature_value <= threshold[p]
                tree_paths[ic]['path']['feature_idx'].append(feature[p])
                tree_paths[ic]['path']['feature_name'].append(feature_name)
                tree_paths[ic]['path']['feature_value'].append(feature_value)
                tree_paths[ic]['path']['threshold'].append(threshold[p])
                tree_paths[ic]['path']['leq_threshold'].append(leq_threshold)

    return(tree_idx, tree_paths)

# parallelisable function for the forest_walker class
def async_regression_tree_walk(tree_idx, instances,
                labels, pred_probas, pred_lodds,
                prior_probas, prior_lodds, delta_lodds,
                # tree_agree_sign_delta,
                feature, threshold, path,
                features, est_wt):

    n_instances = len(labels)
    # object for the results
    tree_paths = [{}] * n_instances

    # case that tree is a single node stump (rare in RF, normal in AdaBoost)
    if len(feature) == 1:
        print(str(tree_idx) + ' is a stump tree')
        for ic in range(n_instances):
            tree_paths[ic] = {'path' : {'feature_idx' : [],
                                        'feature_name' : [],
                                        'feature_value' : [],
                                        'threshold' : [],
                                        'leq_threshold' : []
                                        }
                            }
    # usual case
    else:
        ic = -1 # instance_count
        for p in path:
            if feature[p] < 0: # leaf node
                continue
            if features is None:
                feature_name = None
            else:
                feature_name = features[feature[p]]
            if p == 0: # root node
                ic += 1
                feature_value = instances[ic, [feature[p]]].item(0)
                leq_threshold = feature_value <= threshold[p]
                tree_paths[ic] = {'path' : {'feature_idx' : [feature[p]],
                                            'feature_name' : [feature_name],
                                            'feature_value' : [feature_value],
                                            'threshold' : [threshold[p]],
                                            'leq_threshold' : [leq_threshold]
                                            }
                                }
            else:
                feature_value = instances[ic, [feature[p]]].item(0)
                leq_threshold = feature_value <= threshold[p]
                tree_paths[ic]['path']['feature_idx'].append(feature[p])
                tree_paths[ic]['path']['feature_name'].append(feature_name)
                tree_paths[ic]['path']['feature_value'].append(feature_value)
                tree_paths[ic]['path']['threshold'].append(threshold[p])
                tree_paths[ic]['path']['leq_threshold'].append(leq_threshold)

    for ic in range(n_instances):
        tree_paths[ic].update({'estimator_weight' : abs(est_wt[ic]),
        'pred_class' : int((np.sign(est_wt[ic]) + 1) / 2),
        'pred_value' : est_wt[ic],
        'agree_sign_delta' : np.sign(est_wt[ic]) == np.sign(delta_lodds[ic]),
        'forest_pred_class' : labels[ic],
        'forest_pred_probas' : pred_probas[ic],
        'forest_pred_lodds' : pred_lodds[ic],
        'prior_probas' : prior_probas,
        'prior_lodds' : prior_lodds,
        'delta_lodds' : delta_lodds[ic]
        })

    return(tree_idx, tree_paths)


def async_build_explanation(e_builder,
                sample_instances, sample_labels, forest, forest_walk_mean_elapsed_time,
                paths_lengths_threshold=2, support_paths=0.1, alpha_paths=0.0,
                disc_path_bins=4, disc_path_eqcounts=False,
                score_func=1, weighting='chisq',
                algorithm='greedy_stab',
                merging_bootstraps=0, pruning_bootstraps=0,
                delta=0.1, precis_threshold=0.95, batch_idx=None):
    # these steps make up the CHIRPS process:
    # mine paths for freq patts
    # fp growth mining
    print('as_chirps for batch_idx ' +  str(batch_idx))
    # collect the time taken
    cr_start_time = timeit.default_timer()

    # prime the CHIRPS_runner
    e_builder.sample_instances = sample_instances
    e_builder.sample_labels = sample_labels

    # extract path snippets with highest support/weight
    sp_decrease = support_paths / 10
    while e_builder.patterns is None or len(e_builder.patterns) == 0:
        print('start mining for batch_idx ' + str(batch_idx) + ' with support = ' + str(support_paths))
        e_builder.mine_path_snippets(paths_lengths_threshold, support_paths,
                                disc_path_bins, disc_path_eqcounts)

        # score and sort
        print('found ' + str(len(e_builder.patterns)) + ' patterns from ' + str(len(e_builder.paths)) + ' trees for batch_idx ' +  str(batch_idx))
        support_paths = support_paths - sp_decrease

    print('start score sort for batch_idx ' + str(batch_idx) + ' (' + str(len(e_builder.patterns)) + ') patterns')
    e_builder.score_sort_path_snippets(alpha_paths=alpha_paths,
                                        score_func=score_func,
                                        weighting=weighting)

    # greedily add terms to create rule
    # if len(e_builder.patterns) > 10000:
    #     print('batch_idx ' + str(batch_idx) + ': very long one')
    #     print([cp for i, cp in enumerate(e_builder.patterns) if i < 1000])
    print('start merge rule for batch_idx ' + str(batch_idx) + ' (' + str(len(e_builder.patterns)) + ') patterns')
    e_builder.merge_rule(forest=forest,
                algorithm=algorithm,
                merging_bootstraps=merging_bootstraps,
                pruning_bootstraps=pruning_bootstraps,
                delta=delta,
                precis_threshold=precis_threshold)
    print('merge complete for batch_idx ' + str(batch_idx) + ' (' + str(len(e_builder.patterns)) + ') patterns')

    cr_end_time = timeit.default_timer()
    cr_elapsed_time = cr_end_time - cr_start_time
    cr_elapsed_time = cr_elapsed_time + forest_walk_mean_elapsed_time

    print('start get explainer for batch_idx ' + str(batch_idx))
    exp = e_builder.get_explainer(cr_elapsed_time)
    return(batch_idx, exp)

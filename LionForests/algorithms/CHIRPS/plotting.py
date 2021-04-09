from CHIRPS import in_ipynb
import matplotlib as mpl
if not in_ipynb():
    mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from pandas import Series
from itertools import product
from cycler import cycler
from math import floor, log2

# helper function for plotting conf mat
def plot_confusion_matrix(cm, class_names=None,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, 2)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    if class_names is not None:
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.show()

# helper plot for viewing differences between feature usage
# combining st errs by division: https://chem.libretexts.org/Core/Analytical_Chemistry/Quantifying_Nature/Significant_Digits/Propagation_of_Error
def log_ratio_plot(num, denom, labels, num_err=None, denom_err=None, top=3):
    fig, ax = plt.subplots(figsize=(11, 3))
    log_ratio = np.log(num/denom)

    top_n = np.flip(np.argpartition(log_ratio, -top)[-top:], axis=0)
    bot_n = np.flip(np.argpartition(-log_ratio, -top)[-top:], axis=0)

    lr_top = [log_ratio[i] for i in top_n]
    lr_bot = [log_ratio[i] for i in bot_n]

    if num_err is not None and denom_err is not None:
        ax.stem(top_n, lr_top, linefmt = 'C' + str(1) + ':', markerfmt = 'C' + str(1) + '.')
        ax.stem(bot_n, lr_bot, linefmt = 'C' + str(2) + ':', markerfmt = 'C' + str(2) + '.')

        yerr = 0.434*np.sqrt((num_err/num)**2 + (denom_err/denom)**2)
        ax.errorbar(range(len(labels)), log_ratio, yerr = yerr, fmt='o')
    else:
        ax.stem(range(len(labels)), log_ratio)
        ax.stem(top_n, lr_top, linefmt = 'C' + str(1) + '-', markerfmt = 'C' + str(1) + 'o')
        ax.stem(bot_n, lr_bot, linefmt = 'C' + str(2) + '-', markerfmt = 'C' + str(2) + 'o')

    ax.axhline(0.0, color = 'k', ls = '--')
    ax.annotate('1:1', xy=(-1.0, max(log_ratio) * 0.1))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylabel('log(ratio)')

    plt.show()
    if num_err is not None:
        return(log_ratio, yerr)
    else:
        return(log_ratio)

# helper function for plotting comparison mean path lengths from a forest stats dictionary
def plot_mean_path_lengths(forest_stats, class_names=None):

    classes = [c for c in forest_stats]
    mean_path_lengths = np.zeros(len(classes))
    sd_path_lengths = np.zeros(len(classes))

    for i, c in enumerate(classes):
        mean_path_lengths[i] = forest_stats[c]['m_path_length']
        sd_path_lengths[i] = forest_stats[c]['sd_path_length']

    if class_names:
        classes[:len(class_names)] = class_names

    plt.bar(range(len(classes)),
            mean_path_lengths,
            yerr=sd_path_lengths,
            tick_label=classes)
    plt.title('Mean and St.Dev of Decision Path Length by Class')
    plt.ylabel('Root-to-leaf length (number of nodes)')
    plt.show()

def plot_varimp(forest, features, ordered=False):
    fig, ax = plt.subplots(1, 1, figsize=(11,3))
    x_pos = range(len(features))
    if ordered:
        imp = Series(forest.feature_importances_, index=features).sort_values(ascending=False)
    else:
        imp = Series(forest.feature_importances_, index=features)
    ax.stem(x_pos, imp)
    ax.set_ylabel('Importance %')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(imp.index, rotation='vertical')
    ax.set_title('Variable Importance Plot')
    plt.show()

# helper for plotting varimp
def plot_feature_stats(forest_stats, stat_name,
 class_names=None, features=None):

    plotting_data = [[forest_stats[k][stat_name]] for k in forest_stats.keys() if k != 'all_classes']
    n = len(forest_stats)

    plt.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')

    x_pos = [i for i in range(len(features))]
    fig, ax = plt.subplots(1, 1, figsize=(11,3))

    stms = []
    for s, stat in enumerate(plotting_data):
        linefmt = 'C' + str(s) + '-'
        markerfmt = 'C' + str(s) + 'o'

        stm = ax.stem(x_pos, stat[0], markerfmt=markerfmt, linefmt=linefmt)
        ax.set_xticks(x_pos)
        if features is not None:
            ax.set_xticklabels(features, rotation='vertical')
        ax.set_title(stat_name)
        x_pos = [i + 0.2 for i in x_pos]
        stms.append(stm)

    if class_names is not None:
        plt.legend(stms, class_names)
    else: plt.legend(stms)
    plt.show()

def resize_plot(ax, class_names, set_legend = True):
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    if set_legend:
        ax.legend(title = 'classes', labels = class_names
        , loc='center left', bbox_to_anchor=(1, 0.5))
    return(ax)

def add_maj_match(ax, isolation_pos):
    height_just = floor(ax.get_ylim()[1])
    if isolation_pos is None:
        ax.annotate('target class not matched',
                    xy=(0, height_just * 0.9))
    else:
        ax.axvline(isolation_pos, color = '0.75', ls = '--')
        ax.annotate('target class matched',
                    xy=(isolation_pos + 0.1, height_just * 0.9))
    return(ax)

def add_max_score(ax, score, isolation_pos):
    max_score = np.max(score)
    adj_max_score = np.max(score[isolation_pos:])
    ax.axhline(max_score, color = 'k', ls = '--')

    if adj_max_score < max_score:
        ax.annotate('max score', xy=(0.5, max_score - 0.15))
        ax.axhline(adj_max_score, color = '0.5', ls = '--')
        right_just = floor(ax.get_xlim()[1])
        ax.annotate('max score on correct prediction',
                    xy=(right_just / 2, adj_max_score - 0.15))
    else:
        ax.annotate('max score on correct prediction', xy=(0.5, max_score - 0.15))

def trace_covprecis_plot(ax, measures, measure):
    ax.plot(measures)
    ax.set_ylim(0.0, 1.0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title('Rule ' + measure)
    ax.set_ylabel(measure)
    ax.set_xlabel('number of terms')
    return(ax)

def plot_ig_dist(rule_accumulator, class_names):

    fig, [ax1, ax2, ax3, ax4] = plt.subplots(nrows=4, ncols=1, figsize=(8, 8), dpi=80
                                    , facecolor='w', edgecolor='k')

    ax1.plot(rule_accumulator.cum_info_gain)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.axhline(rule_accumulator.max_ent, color = 'k', ls = '--')
    ax1.axhline(rule_accumulator.model_info_gain, color = '0.5', ls = '--')
    ax1.axhline(rule_accumulator.prior_info, color = '0.75', ls = '--')
    ax1.set_title('IG per term added to rule')
    ax1.set_ylabel('Cum. Info. Gain')
    right_just = floor(ax1.get_xlim()[1])
    ax1.annotate('max entropy for ' + str(rule_accumulator.n_classes) + ' class problem',
                xy=(0.5, rule_accumulator.max_ent - 0.15))
    ax1.annotate('model info gain',
                xy=(right_just / 2, rule_accumulator.model_info_gain - 0.15))
    ax1.annotate('prior information',
                xy=(right_just - 2, rule_accumulator.prior_info + 0.05))

    ax2.plot(rule_accumulator.pri_and_post)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_ylim(0.0, 1.0)
    ax2 = add_maj_match(ax2, rule_accumulator.isolation_pos)
    ax2.set_title('Posterior distributions (Precision wrt Class)')
    ax2.set_ylabel('P(y = class)')

    ax3.plot(np.log(rule_accumulator.pri_and_post_counts + 1))
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax3 = add_maj_match(ax3, rule_accumulator.isolation_pos)
    ax3.set_title('Number of instances')
    ax3.set_ylabel('log(counts)')

    ax4.plot(np.log(rule_accumulator.pri_and_post_lift + 1))
    ax4.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax4 = add_maj_match(ax4, rule_accumulator.isolation_pos)
    ax4.set_title('Lift wrt Class')
    ax4.set_ylabel('log(lift)')

    ax1 = resize_plot(ax1, class_names, set_legend=False)
    ax2 = resize_plot(ax2, class_names)
    ax3 = resize_plot(ax3, class_names)
    ax4 = resize_plot(ax4, class_names)

    fig.suptitle('Rule trace through scoring sample', fontsize=12)
    fig.tight_layout(rect=[0, 0.0, 1, 0.95])

    plt.show()

def plot_coverage_precision(rule_accumulator, class_names):
    # plot the rule trace based on coverage and precision
    fig, [ax1, ax2, ax3, ax4] = plt.subplots(nrows=4, ncols=1, figsize=(8, 8), dpi=80
                                    , facecolor='w', edgecolor='k')

    ax1 = trace_covprecis_plot(ax1, rule_accumulator.coverage, 'Coverage')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = trace_covprecis_plot(ax2, rule_accumulator.pri_and_post_recall, 'Coverage wrt Class (Recall)')
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = add_maj_match(ax2, rule_accumulator.isolation_pos)
    ax2 = resize_plot(ax2, class_names)
    ax3 = trace_covprecis_plot(ax3, rule_accumulator.pri_and_post_accuracy, 'Accuracy wrt Class')
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax3 = add_maj_match(ax3, rule_accumulator.isolation_pos)
    ax3 = resize_plot(ax3, class_names)
    ax4 = trace_covprecis_plot(ax4, rule_accumulator.pri_and_post_f1, 'F1 wrt Class')
    ax4.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax4 = add_maj_match(ax4, rule_accumulator.isolation_pos)
    ax4 = resize_plot(ax4, class_names)

    fig.suptitle('Rule trace through scoring sample', fontsize=12)
    fig.tight_layout(rect=[0, 0.0, 1, 0.95])

    plt.show()

# used to be part of rule acc, now CHIRPS_runner. will need updating
def score_rule(self, alpha=0.5):
    target_precision = [p[self.target_class] for p in self.posterior]
    target_stability = [s[self.target_class] for s in self.posterior]
    target_recall = [r[self.target_class] for r in self.recall]
    target_f1 = [f[self.target_class] for f in self.f1]
    target_accuracy = [a[self.target_class] for a in self.accuracy]
    target_prf = [[p, s, r, f, a] for p, s, r, f, a in zip(target_precision, target_stability, target_recall, target_f1, target_accuracy)]

    target_cardinality = [i for i in range(len(target_precision))]

    lf = lambda x: math.log2(x + 1)
    score_fun1 = lambda f, crd, alp: lf(f * crd * alp / (1.0 + ((1 - alp) * crd**2)))
    score_fun2 = lambda a, crd, alp: lf(a * crd * alp / (1.0 + ((1 - alp) * crd**2)))

    score1 = [s for s in map(score_fun1, target_f1, target_cardinality, [alpha] * len(target_cardinality))]
    score2 = [s for s in map(score_fun2, target_accuracy, target_cardinality, [alpha] * len(target_cardinality))]

    return(target_prf, score1, score2)


# plot the rule trace based on entropy and posteriors
def plot_rule_scores(rule_accumulator, class_names, alpha_scores=0.5):

    fig, [ax1, ax2, ax3] = plt.subplots(nrows=3, ncols=1, figsize=(8, 6), dpi=80
                                    , facecolor='w', edgecolor='k')

    prf, score1, score2 = rule_accumulator.score_rule(alpha_scores)
    ax1 = trace_covprecis_plot(ax1, prf, 'Measures wrt Target')
    ax1 = add_maj_match(ax1, rule_accumulator.isolation_pos)
    ax1 = resize_plot(ax1, ['Precision', 'Recall', 'F1', 'Accuracy'])

    ax2 = trace_covprecis_plot(ax2, score1, 'Score Function 1')
    ax2 = add_maj_match(ax2, rule_accumulator.isolation_pos)
    ax2 = add_max_score(ax2, score1, rule_accumulator.isolation_pos)

    ax3 = trace_covprecis_plot(ax3, score2, 'Score Function 2')
    ax3 = add_maj_match(ax3, rule_accumulator.isolation_pos)
    ax3 = add_max_score(ax3, score2, rule_accumulator.isolation_pos)

    fig.suptitle('Rule trace through scoring sample', fontsize=12)
    fig.tight_layout(rect=[0, 0.0, 1, 0.95])

    plt.show()

# plot the rule trace based on entropy and posteriors
def rule_profile_plots(rule_accumulator, class_names, alpha_scores=0.5, ig=True, cp=True, rs=True):

    if ig:
        plot_ig_dist(rule_accumulator, class_names)
    if cp:
        plot_coverage_precision(rule_accumulator, class_names)
    if rs:
        plot_rule_scores(rule_accumulator, class_names, alpha_scores)

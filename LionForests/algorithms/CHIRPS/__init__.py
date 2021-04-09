import math
import numpy as np
from pathlib import Path as pth
from os import makedirs as mkdir
from scipy.stats import chi2_contingency, entropy

# helper function determines if we're in a jup notebook
def in_ipynb():
    try:
        cfg = get_ipython().config
        if len(cfg.keys()) > 0:
            if list(cfg.keys())[0]  == 'IPKernelApp':
                return(True)
            else:
                return(False)
        else:
            return(False)
    except NameError:
        return(False)

def p_count_corrected(arr, classes, weights=None):
    # quick out if nothing is passed in
    if len(arr) == 0:
        return({'labels' : classes,
        'counts' : np.zeros(len(classes)),
        'p_counts' : np.zeros(len(classes))})

    # otherwise
    # initialise weights to ones, accept a vector of weights, or extract per row weights from a 2D array
    if weights is None:
        weights = np.ones(len(arr))
    else:
        weights = np.array(weights)
        if len(weights.shape) == 1:
            pass
        else:
            n_weights = np.shape(weights)[0]
            weights = weights[range(n_weights), arr] # this will fail if there is no second dimension
            weights = weights / (n_weights * len(classes)) # this is the formula for SAMME.R and scikit


    c = np.bincount(arr, weights=np.array(weights))
    
    # correct for any classes at the end of the sequence not represented e.g. a string of 0, 1, but there are 0, 1 and 2 classes
    # the bincount function cuts off the last unrepresented class
    unrep = abs(len(classes) - len(c)) #added abs
    c = np.append(c, np.zeros(unrep))

    pc = c / weights.sum()

    return({'labels' : classes,
    'counts' : c,
    'p_counts' : pc})

def chisq_indep_test(counts, prior_counts):
    if type(counts) == list:
        counts = np.array(counts)
    observed = np.array((counts, prior_counts))
    if len(observed.shape)==1:
        return [0, 0, 0]
    if counts.sum() > 0: # previous_counts.sum() == 0 is impossible
        chisq = chi2_contingency(observed=observed[:, np.where(observed.sum(axis=0) != 0)], correction=True)[0:3]
    else:
        r, c = observed.shape
        chisq = (np.nan, np.nan, (r - 1) * (c - 1))
    return(chisq)

def entropy_corrected(p, q=None):
    if q is None:
        return(entropy(p))
    alpha = np.finfo(dtype='float32').eps
    p_smooth = np.random.uniform(size=len(p))
    q_smooth = np.random.uniform(size=len(p)) # convex smooth idea https://mathoverflow.net/questions/72668/how-to-compute-kl-divergence-when-pmf-contains-0s
    p_smoothed = p_smooth * alpha + np.array(p) * (1-alpha)
    if len(q) == len(q_smooth): 
        q_smoothed = q_smooth * alpha + np.array(q) * (1-alpha)
    else:
        q_smoothed = [0] * len(p_smoothed)
    return(entropy(p_smoothed, q_smoothed))

def contingency_test(p, q, statistic = 'chisq'):
    if statistic == 'chisq':
        return(math.sqrt(chisq_indep_test(p, q)[0]))
    elif statistic == 'kldiv':
        return(entropy_corrected(p, q))

# create a directory if doesn't exist
def if_nexists_make_dir(save_path):
    if not pth(save_path).is_dir():
        mkdir(save_path)

# create a file if doesn't exist
def if_nexists_make_file(save_path, init_text='None'):
    if not pth(save_path).is_file():
        f = open(save_path, 'w+')
        f.write(init_text)
        f.close()

def confidence_weight(proba, what='conf_weight'):
    proba = np.array(proba)
    if what=='proba':
        return(proba)
    # Displace zero probabilities so the log is defined.
    # Also fix negative elements which may occur with
    # negative sample weights.
    proba[proba < np.finfo(proba.dtype).eps] = np.finfo(proba.dtype).eps
    log_proba = np.log(proba)
    if what=='log_proba':
        return(log_proba)
    n_classes = len(proba[0])
    # conf_weight
    return( (n_classes - 1.) * (log_proba - (1. / n_classes)
                              * log_proba.sum(axis=1)[:, np.newaxis]) )

def o_print(text, verbose=True):
    if verbose == True:
        print(text)

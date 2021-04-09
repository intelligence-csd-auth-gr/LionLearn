import io
import zipfile
import urllib
import numpy as np
import pandas as pd
from datetime import datetime
import math
import julian
import re
from sklearn.impute import SimpleImputer
import unicodedata

# adult from source
if True:
    '''
    random_state = 123
    var_names = ['age'
           , 'workclass'
           , 'lfnlwgt'
           , 'education'
           , 'educationnum'
           , 'maritalstatus'
           , 'occupation'
           , 'relationship'
           , 'race'
           , 'sex'
           , 'lcapitalgain'
           , 'lcapitalloss'
           , 'hoursperweek'
           , 'nativecountry'
           , 'income']

    vars_types = ['continuous'
           , 'nominal'
           , 'continuous'
           , 'nominal'
           , 'continuous'
           , 'nominal'
           , 'nominal'
           , 'nominal'
           , 'nominal'
           , 'nominal'
           , 'continuous'
           , 'continuous'
           , 'continuous'
           , 'nominal'
           , 'nominal']

    adult_train_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    adult_test_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'

    adult_train_bytes = urllib.request.urlopen(adult_train_url)
    adult_test_bytes = urllib.request.urlopen(adult_test_url)
    adult_train = pd.read_csv(adult_train_bytes, header=None, names=var_names)
    adult_test = pd.read_csv(adult_test_bytes, header=None, names=var_names)
    adult_test.drop(index=0, axis=0, inplace=True) # there's a stupid line at the top. skiprows doesn't deal with it

    # combine the two datasets and split them later with standard code
    frames = [adult_train, adult_test]
    adult = pd.concat(frames)

    # some tidying required
    adult.income = adult.income.str.replace('.', '')
    for f, t in zip(var_names, vars_types):
        if t == 'continuous':
            adult[f] = adult[f].astype('int32')
        else:
            adult[f] = adult[f].str.replace(' ', '')

    # change question mark character to 'Unknown'
    qm_to_unk = lambda w: 'Unknown' if w == '?' else w

    # duplication / tidy of a country entry
    tt_fix = lambda w: 'Trinidad and Tobago' if w == 'Trinadad&Tobago' else w

    adult['workclass'] = adult.workclass.apply(qm_to_unk)
    adult['nativecountry'] = adult.nativecountry.apply(qm_to_unk)
    adult['nativecountry'] = adult.nativecountry.apply(tt_fix)

    # make these numeric values a bit closer to normal distr.
    adult['lcapitalgain'] = np.log(adult['lcapitalgain'] + abs(adult['lcapitalgain'].min()) + 1)
    adult['lcapitalloss'] = np.log(adult['lcapitalloss'] + abs(adult['lcapitalloss'].min()) + 1)
    adult['lfnlwgt'] = np.log(adult['lfnlwgt'] + abs(adult['lfnlwgt'].min()) + 1)

    # create a small set that is easier to play with on a laptop
    adult_samp = adult.sample(frac=0.5, random_state=random_state).reset_index()
    adult_samp.drop(labels='index', axis=1, inplace=True)

    # create a small set that is easier to play with on a laptop
    adult_small_samp = adult.sample(frac=0.05, random_state=random_state).reset_index()
    adult_small_samp.drop(labels='index', axis=1, inplace=True)

    # save
    adult.to_csv('CHIRPS\\datafiles\\adult.csv.gz', index=False, compression='gzip')
    adult_samp.to_csv('CHIRPS\\datafiles\\adult_samp.csv.gz', index=False, compression='gzip')
    adult_small_samp.to_csv('CHIRPS\\datafiles\\adult_small_samp.csv.gz', index=False, compression='gzip')
    '''

# bankmark from source
if True:
    '''
    # random seed for train test split and sampling
    random_state = 123

    def import_bankmark_file(file):
        archive = zipfile.ZipFile('CHIRPS/source_datafiles/bank-additional-full.zip', 'r')
        lines = archive.read(file).decode("utf-8").split('\r\n')
        lines = [lines[i].replace('\"', '').split(';') for i in range(1, len(lines))]
        lines.pop() # last item is empty
        archive.close()
        return(lines)

    test_lines = import_bankmark_file('bank-additional.csv')
    train_lines = import_bankmark_file('bank-additional-full.csv')
    all_lines = train_lines + test_lines

    names = ['age','job','marital','education','default','housing','loan','contact','month',
             'day_of_week','duration','campaign','pdays','previous','poutcome','emp.var.rate',
             'cons.price.idx','cons.conf.idx','euribor3m','nr.employed','y']

    vtypes = {'age' : np.uint8, 'job' : object, 'marital' : object, 'education' : object, 'default' : object,
                  'housing' : object, 'loan' : object, 'contact' : object, 'month' : object, 'day_of_week' : object,
                  'duration' : np.uint16, 'campaign' : np.uint8, 'pdays' : np.uint8, 'previous' : np.uint8,
                  'poutcome' : object, 'emp.var.rate' : np.float16, 'cons.price.idx' : np.float16,
                  'cons.conf.idx' : np.float16, 'euribor3m' : np.float16, 'nr.employed' : np.float16, 'y' : object}

    bankmark = pd.DataFrame(all_lines, columns=names)
    bankmark = bankmark.astype(dtype=vtypes)

    # save
    bankmark.to_csv('CHIRPS\\datafiles\\bankmark.csv.gz', index=False, compression='gzip')

    # create small set that is easier to play with on a laptop
    samp = bankmark.sample(frac=0.05, random_state=random_state).reset_index()
    samp.drop(labels='index', axis=1, inplace=True)
    samp.to_csv('CHIRPS\\datafiles\\bankmark_samp.csv.gz', index=False, compression='gzip')
    '''

# car form source
if True:
    '''
    var_names = ['buying'
                , 'maint'
                , 'doors'
                , 'persons'
                , 'lug_boot'
                , 'safety'
                , 'acceptability']

    target_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data'

    car_bytes = urllib.request.urlopen(target_url)
    car = pd.read_csv(car_bytes, header=None, names=var_names)
    # recode to a 2 class subproblems
    car.acceptability.loc[car.acceptability != 'unacc'] = 'acc'

    car.to_csv('CHIRPS\\datafiles\\car.csv.gz', index=False, compression='gzip')
    '''

# cardio
if True:
    '''
    target_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00193/CTG.xls'
    cardio_bytes = urllib.request.urlopen(target_url)
    cardio = pd.read_excel(cardio_bytes,
                            sheet_name='Data',
                            header=1)

    var_names_raw = ['LB'
                , 'AC.1'
                , 'FM.1'
                , 'UC.1'
                , 'DL.1'
                , 'DS.1'
                , 'DP.1'
                , 'ASTV'
                , 'MSTV'
                , 'ALTV'
                , 'MLTV'
                , 'Width'
                , 'Min'
                , 'Max'
                , 'Nmax'
                , 'Nzeros'
                , 'Mode'
                , 'Mean'
                , 'Median'
                , 'Variance'
                , 'Tendency'
                #, 'CLASS' Exclude as this is an alternative target
                , 'NSP']

    cardio = cardio.loc[:, var_names_raw]

    var_names = ['LB'
                , 'AC'
                , 'FM'
                , 'UC'
                , 'DL'
                , 'DS'
                , 'DP'
                , 'ASTV'
                , 'MSTV'
                , 'ALTV'
                , 'MLTV'
                , 'Width'
                , 'Min'
                , 'Max'
                , 'Nmax'
                , 'Nzeros'
                , 'Mode'
                , 'Mean'
                , 'Median'
                , 'Variance'
                , 'Tendency'
                #, 'CLASS'
                , 'NSP']

    cardio.columns = var_names

    # remove the last three rows that are aggragates in the raw data file
    cardio = cardio.loc[~cardio['LB'].isna(), :]

    # re-code NSP and delete class variable
    NSP = pd.Series(['N'] * cardio.shape[0])
    NSP.loc[cardio.NSP.values == 2] = 'S'
    NSP.loc[cardio.NSP.values == 3] = 'P'
    cardio.NSP = NSP

    # save
    cardio.to_csv('CHIRPS\\datafiles\\cardio.csv.gz', index=False, compression='gzip')
    '''

# credit from source
if True:
    '''
    var_names = ['A1'
                , 'A2'
                , 'A3'
                , 'A4'
                , 'A5'
                , 'A6'
                , 'A7'
                , 'A8'
                , 'A9'
                , 'A10'
                , 'A11'
                , 'A12'
                , 'A13'
                , 'A14'
                , 'A15'
                , 'A16']

    vars_types = ['nominal'
        , 'continuous'
        , 'continuous'
        , 'nominal'
        , 'nominal'
        , 'nominal'
        , 'nominal'
        , 'continuous'
        , 'nominal'
        , 'nominal'
        , 'continuous'
        , 'nominal'
        , 'nominal'
        , 'continuous'
        , 'continuous'
        , 'nominal'
    ]

    target_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data'

    credit_bytes = urllib.request.urlopen(target_url)
    credit = pd.read_csv(credit_bytes,
                         header=None,
                         delimiter=',',
                         index_col=False,
                         names=var_names,
                         na_values = '?')

    # re-code rating class variable
    A16 = pd.Series(['plus'] * credit.shape[0])
    A16.loc[credit.A16.values == '-'] = 'minus'
    credit.A16 = A16

    # deal with some missing data
    for v, t in zip(var_names, vars_types):
        if t == 'nominal':
            credit[v] = credit[v].fillna('u')
        else:
            credit[v] = credit[v].fillna(credit[v].mean())

    credit.to_csv('CHIRPS\\datafiles\\credit.csv.gz', index=False, compression='gzip')
    '''

# german from source
if True:
    '''
    var_names = ['chk'
                , 'dur'
                , 'crhis'
                , 'pps'
                , 'amt'
                , 'svng'
                , 'emp'
                , 'rate'
                , 'pers'
                , 'debt'
                , 'res'
                , 'prop'
                , 'age'
                , 'plans'
                , 'hous'
                , 'creds'
                , 'job'
                , 'deps'
                , 'tel'
                , 'foreign'
                , 'rating']

    target_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data'

    german_bytes = urllib.request.urlopen(target_url)
    german = pd.read_csv(german_bytes,
                         header=None,
                         delimiter=' ',
                         index_col=False,
                         names=var_names)

    # re-code rating class variable
    rating = pd.Series(['good'] * german.count()[0])
    rating.loc[german.rating == 2] = 'bad'
    german.rating = rating

    german.to_csv('CHIRPS\\datafiles\\german.csv.gz', index=False, compression='gzip')
    '''

# lending from source
if True:
    '''
    # lending from source
    # download the files accepted_2007_to_2018Q2.csv.gz from Kaggle
    # https://www.kaggle.com/wendykan/lending-club-loan-data
    # there is also a rejected_2007_to_2018Q2.csv.gz file but we are interested in who paid off a loan
    random_state=123
    lending = pd.read_csv('CHIRPS\\source_datafiles\\accepted_2007_to_2018Q2.csv.gz',
                          compression='gzip', low_memory=False)     # low_memory=False prevents mixed data types in the DataFrame

    # Just looking at loans that met the policy and were either fully paid or charged off (finally defaulted)
    lending = lending.loc[lending['loan_status'].isin(['Fully Paid', 'Charged Off'])]
    lending.reset_index(inplace=True, drop=True)

    # data set is wide. What can be done to reduce it? lots to clean up and some useful transforms

    # drop cols with only one distinct value
    drop_list = []
    for col in lending.columns:
        if lending[col].nunique() == 1:
            drop_list.append(col)

    lending.drop(labels=drop_list, axis=1, inplace=True)

    # drop cols with excessively high missing amounts
    drop_list = []
    for col in lending.columns:
        if lending[col].notnull().sum() / lending.shape[0] < 0.5:
            drop_list.append(col)

    lending.drop(labels=drop_list, axis=1, inplace=True)

    # more noisy columns
    lending.drop(labels=['id', 'title', 'emp_title', 'url',
                         'application_type', 'acc_now_delinq',
                         'num_tl_120dpd_2m', 'num_tl_30dpd'],
                 axis=1, inplace=True)

    # highly correlated with the class
    lending.drop(labels=['collection_recovery_fee', 'debt_settlement_flag', 'recoveries'], axis=1, inplace=True)

    # no need for an upper and lower fico, they are perfectly correlated. Take the mean of each pair.
    fic = ['fico_range_low', 'fico_range_high']
    lastfic = ['last_fico_range_low', 'last_fico_range_high']
    lending['fico'] = lending[fic].mean(axis=1)
    lending['last_fico'] = lending[lastfic].mean(axis=1)
    lending.drop(labels=fic + lastfic, axis=1, inplace=True)

    # slightly more informative coding of these vars that are mostly correlated with loan amnt and/or high skew
    lending['non_funded_score'] = np.log(lending['loan_amnt'] + 1 - lending['funded_amnt'])
    lending['non_funded_inv_score'] = np.log(lending['loan_amnt'] + 1 - lending['funded_amnt_inv'])
    lending['adj_log_dti'] = np.log(lending['dti'] + abs(lending['dti'].min()) + 1)
    lending['log_inc'] = np.log(lending['annual_inc'] + abs(lending['annual_inc'].min()) + 1)
    lending.drop(['funded_amnt', 'funded_amnt_inv', 'dti', 'annual_inc'], axis=1, inplace=True)

    # julian dates are better, nice continuous input.
    jul_conv = lambda x : np.nan if x == 'Unknown' else julian.to_jd(datetime.strptime(x, '%b-%Y'))
    for date_col in ['issue_d' , 'last_credit_pull_d', 'earliest_cr_line', 'last_pymnt_d']:
        dc = pd.Series(['Unknown'] * lending.shape[0])
        dc.loc[~lending[date_col].isnull().values] = lending[date_col].loc[~lending[date_col].isnull().values]
        lending[date_col] = dc.map(jul_conv)

    # this one feature has just a tiny number of missing. OK to impute.
    lending['last_credit_pull_d'] = lending.last_credit_pull_d.fillna(lending.last_credit_pull_d.mean())

    # convert 'term' to int
    lending['term'] = lending['term'].apply(lambda s:np.float(s[1:3])) # There's an extra space in the data for some reason

    # convert sub-grade to float and remove grade
    grade_dict = {'A':0.0, 'B':1.0, 'C':2.0, 'D':3.0, 'E':4.0, 'F':5.0, 'G':6.0}
    grade_to_float = lambda s : 5 * grade_dict[s[0]] + np.float(s[1]) - 1

    lending['sub_grade'] = lending['sub_grade'].map(grade_to_float)
    lending.drop(labels='grade', axis=1, inplace=True)

    # convert emp_length - assume missing and < 0 is no job or only very recent started job
    # emp length is only significant for values of 0 or not 0
    emp_conv = lambda e : 'U' if pd.isnull(e) or e[0] == '<' else 'E'
    lending['emp'] = lending['emp_length'].map(emp_conv)
    lending.drop(labels='emp_length', axis=1, inplace=True)

    # tidy up some very minor class codes in home ownership
    lending['home_ownership'] = lending['home_ownership'].map(lambda h: 'OTHER' if h in ['ANY', 'NONE'] else h)

    # there is a number of rows that have missing data for many variables in a block pattern -
    # these are probably useless because missingness goes across so many variables
    # it might be possible to save them to a different set and create a separate model on them

    # another approach is to fill them with an arbitrary data point (means, zeros, whatever)
    # and add a new feature that is binary for whether this row had missing data
    # this will give the model something to adjust/correlate/associate with if these rows turn out to add noise

    # 'avg_cur_bal is a template for block missingness
    # will introduce a missing indicator column based on this
    # then fillna with zeros and finally filter out some unsalvageable really rows
    lending['block_missingness'] = lending['avg_cur_bal'].isnull() * 1.0
    # rows where last_pymnt_d is zero are just a mess, get them outa here. all other nans get changed to zero
    lending = lending.fillna(0)
    lending = lending[lending.last_pymnt_d != 0]
    # and a final reindex
    lending.reset_index(inplace=True, drop=True)

    # and rearrange so class_col is at the end
    class_col = 'loan_status'
    pos = np.where(lending.columns == class_col)[0][0]
    var_names = list(lending.columns[:pos]) + list(lending.columns[pos + 1:]) + list(lending.columns[pos:pos + 1])
    lending = lending[var_names]

    # create a small set that is easier to play with on a laptop
    lend_samp = lending.sample(frac=0.1, random_state=random_state).reset_index()
    lend_samp.drop(labels='index', axis=1, inplace=True)
    lend_small_samp = lending.sample(frac=0.01, random_state=random_state).reset_index()
    lend_small_samp.drop(labels='index', axis=1, inplace=True)
    lend_tiny_samp = lending.sample(frac=0.0025, random_state=random_state).reset_index()
    lend_tiny_samp.drop(labels='index', axis=1, inplace=True)

    # save
    lending.to_csv('CHIRPS\\datafiles\\lending.csv.gz', index=False, compression='gzip')
    lend_samp.to_csv('CHIRPS\\datafiles\\lending_samp.csv.gz', index=False, compression='gzip')
    lend_small_samp.to_csv('CHIRPS\\datafiles\\lending_small_samp.csv.gz', index=False, compression='gzip')
    lend_tiny_samp.to_csv('CHIRPS\\datafiles\\lending_tiny_samp.csv.gz', index=False, compression='gzip')
    '''

# nursery
if True:
    '''
    random_state = 123
    var_names = ['parents'
                   , 'has_nurs'
                   , 'form'
                   , 'children'
                   , 'housing'
                   , 'finance'
                   , 'social'
                   , 'health'
                   , 'decision']

    target_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data'
    nursery_bytes = urllib.request.urlopen(target_url)
    nursery = pd.read_csv(nursery_bytes, header=None, names=var_names)

    # clean up: filter single row where class == 2
    nursery = nursery[nursery.decision != 'recommend']
    nursery.to_csv('CHIRPS\\datafiles\\nursery.csv.gz', index=False, compression='gzip')

    samp = nursery.sample(frac=0.2, random_state=random_state).reset_index()
    samp.drop(labels='index', axis=1, inplace=True)
    samp.to_csv('CHIRPS\\datafiles\\nursery_samp.csv.gz', index=False, compression='gzip')
    '''

# rcdv
if True:
    '''
    # original datasets here: https://www.icpsr.umich.edu/icpsrweb/NACJD/studies/8987/datadocumentation
    # files need to be processed before import. fixed width text
    # needs to be split out into variables according to the code book in the available documentation

    # random seed for train test split and sampling
    random_state = 123

    myzip = 'CHIRPS\\source_datafiles\\rcdv_processed.zip'
    myfile = 'rcdv_processed.xlsx'
    zf = zipfile.ZipFile(myzip)
    zb = zf.read(myfile)
    rcdv = pd.read_excel(io.BytesIO(zb)
                            , sheet_name='1978'
                            , header=0)
    # merging two sheets
    rcdv = rcdv.append(pd.read_excel(io.BytesIO(zb)
                            , sheet_name='1980'
                            , header=0))
    zf.close()

    # tidy index and drop col1
    rcdv.reset_index(drop=True, inplace=True)
    rcdv.drop(labels=rcdv.columns[0], axis=1, inplace=True)

    # give last column a more meaningful name - it indicates rows that had missing data
    rcdv.columns = ['missingness' if vn == 'file' else vn for vn in rcdv.columns]

    # recode priors, all that were set to -9 were missing
    # and it is logged in the file variable (3 = missing data indicator)
    # missingness also affect alchy and junky variables
    rcdv['priors'] = rcdv['priors'].map(lambda x: 0 if x == -9 else x)
    rcdv['missingness'] = rcdv['missingness'].map(lambda x: 1 if x == 3 else 0)

    # remove Time column. It is for survival analysis. Correlates exactly with recid.
    del rcdv['time']

    # put recid to the end
    recid_pos = np.where(rcdv.columns == 'recid')[0][0]
    var_names = list(rcdv.columns[0:recid_pos]) + list(rcdv.columns[recid_pos + 1: len(rcdv.columns)]) + ['recid']
    rcdv = rcdv[var_names]

    # convert class column to a categorical
    rcdv['recid'] = rcdv['recid'].transform(lambda x: 'Y' if x == 1 else 'N')

    # save it out
    rcdv.to_csv('CHIRPS\\datafiles\\rcdv.csv.gz', index=False, compression='gzip')

    # create small set that is easier to play with on a laptop
    samp = rcdv.sample(frac=0.1, random_state=random_state)
    samp.reset_index(drop=True, inplace=True)
    samp.to_csv('CHIRPS\\datafiles\\rcdv_samp.csv.gz', index=False, compression='gzip')
    '''

if True:
    '''
    random_state=123
    file = 'usoc_wv23.csv'
    # usoc = import_usoc_file('usoc_all.csv')
    archive = zipfile.ZipFile('CHIRPS/source_datafiles_proprietary/usoc.zip', 'r')
    lines = archive.read(file).decode("utf-8").split('\r\n')
    archive.close()
    # convert strings to var lists
    lines = [lines[i].replace('\ufeff', '').split(',') for i in range(len(lines))]
    names = lines[0]
    lines = lines[1:]
    usoc = pd.DataFrame(lines, columns=names)
    usoc.drop(columns=['pidp', 'pid',
                       'b_hidp', 'b_pno', 'b_splitnum',
                       'c_hidp', 'c_pno', 'c_splitnum',
                       'dord', 'dory', 'dorm',
                       'height', 'resnhi', 'bmi', 'age',
                       'ehtch', 'ehtm', 'ehtft', 'ehtin',
                       'weight', 'ewtkg', 'ewtst', 'ewtl',
                       'waist1', 'waist2', 'waist3',
                       'ag16g10', 'ag16g20', 'vpstimehh',
                       'vpstimemm', 'strtnurhh', 'strtnurmm',
                       'psu', 'strata', 'indns91_lw', 'indns01_lw',
                       'indnsub_lw', 'indnsbh_xw', 'indnsub_xw'], inplace=True)
    # remove rows with properly missing values
    missing = set(np.array([usoc.index[usoc[c] == ' '].tolist() for c in usoc.columns if len(usoc.index[usoc[c] == ' ']) > 0]).reshape(1,-1)[0])
    usoc.drop(index=missing, inplace=True)

    # vars with . somewhere are floats
    vtypes = {n : np.float16 if any(['.' in value for value in usoc[n]]) else np.object for n in usoc.columns}
    usoc = usoc.astype(dtype=vtypes)
    # nearly all vars with ten or fewer uniques are nominal
    vtypes = {n : np.int16 if not np.issubdtype(usoc[n], np.float) and len(set([value for value in usoc[n]])) > 10 else usoc[n].dtype for n in usoc.columns}
    # these are nominal with larger numbers of uniques - have to do manually based on dictionary
    vtypes.update({'nuroutc' : np.object, 'lfout' : np.object,
                  'elig' : np.object, 'ethnic' : np.object,
                  'hhtype_dv' : np.object, 'jbstat' : np.object,
                  'mlstat' : np.object, 'marstat' : np.object,
                  'jbnssec8_dv' : np.object, 'jlnssec8_dv' : np.object})
    usoc = usoc.astype(dtype=vtypes)

    usoc.reset_index(inplace=True)
    usoc.to_csv('CHIRPS\\datafiles_proprietary\\usoc.csv.gz', index=False, compression='gzip')
    random_state = 123
    samp = usoc.sample(frac=0.1, random_state=random_state)
    samp.reset_index(drop=True, inplace=True)
    samp.to_csv('CHIRPS\\datafiles_proprietary\\usoc_samp.csv.gz', index=False, compression='gzip')

    # convert the general health likert to a simple class
    usoc2 = usoc[np.logical_and(np.logical_and(usoc.scghq1_dv >=0, usoc.bmival >=0), usoc.wstval >=0)]
    usoc2.reset_index(inplace=True)
    mh = np.array(['neutral'] * len(usoc2.scghq1_dv))
    mh[usoc2.scghq1_dv < 7.0] = 'poor'
    mh[usoc2.scghq1_dv > 13.0] = 'good'
    usoc2 = usoc2.assign(mh = pd.Series(mh, index = usoc2.index))
    usoc2.drop(columns='scghq1_dv', inplace=True)
    usoc2.to_csv('CHIRPS\\datafiles_proprietary\\usoc2.csv.gz', index=False, compression='gzip')
    random_state = 123
    samp = usoc2.sample(frac=0.1, random_state=random_state)
    samp.reset_index(drop=True, inplace=True)
    samp.to_csv('CHIRPS\\datafiles_proprietary\\usoc2_samp.csv.gz', index=False, compression='gzip')
    '''

if True:
    '''
    random_state=123
    file = 'hospital_readmission.csv'
    archive = zipfile.ZipFile('CHIRPS/source_datafiles/hospital_readmission.zip', 'r')
    lines = archive.read(file).decode("utf-8").split('\n')
    archive.close()
    # convert strings to var lists. Replace true/false with binary
    lines = [lines[i].replace('False', str(0)).replace('True', str(1)).split(',') for i in range(len(lines))]
    # the last line is corrupt - must have been a newline character at the end
    lines.pop()
    # get the header and the lines
    names = lines[0]
    lines = lines[1:]
    readmission = pd.DataFrame(lines, columns=names)
    readmission = readmission.astype(dtype=np.int16)
    readmission['readmitted'] = pd.Series(['T' if i == 1 else 'F' for i in readmission.readmitted])

    var_names = readmission.columns.to_list()
    var_names = [vn for vn in var_names if vn != 'readmitted']
    var_names.append('readmitted')
    readmission = readmission[var_names] # put the class col at the end

    readmission.to_csv('CHIRPS\\datafiles\\readmit.csv.gz', index=False, compression='gzip')
    random_state = 123
    samp = readmission.sample(frac=0.1, random_state=random_state)
    samp.reset_index(drop=True, inplace=True)
    samp.to_csv('CHIRPS\\datafiles\\readmit_samp.csv.gz', index=False, compression='gzip')
    '''

if True:
    '''
    file = '2014.csv'
    archive = zipfile.ZipFile('CHIRPS/source_datafiles/mhtech.zip', 'r')
    lines = archive.read(file).decode("utf-8").split('\n')
    archive.close()

    # convert strings to var lists - deal with some free text issues
    lines = [lines[i].replace("Bahamas, The", "Bahamas").replace("male, unsure", "male unsure").split(',') for i in range(len(lines))]
    names = [nm.replace('"', '') for nm in lines[0]]
    lines = lines[1:]
    lines.pop() # empty final row
    mhtech = pd.DataFrame(lines, columns=names)

    # not useful, turns out state only valid for USA
    mhtech.drop(['Timestamp', 'state'], 1, inplace=True)

    # rename columns for ease
    ren = {old : new for old, new in zip(mhtech.columns, [lc.lower() for lc in mhtech.columns])}
    mhtech.rename(columns=ren, inplace=True)

    # empty comments
    mhtech['comments'] = data.comments.fillna('No comment')

    # standardise some other responses, getting rid of 'apos in "Don't know"
    for c in mhtech.columns:
        mhtech[c] = mhtech[c].str.replace("Don't know", 'Not sure')
        mhtech[c] = mhtech[c].str.replace('"', '')
        mhtech[c] = mhtech[c].str.replace('Maybe', 'Not sure')
        mhtech[c] = mhtech[c].str.replace('not sure', 'Not sure')
        mhtech[c] = mhtech[c].str.replace('Some of them', 'Yes')
        mhtech[c] = mhtech[c].str.replace('NA', 'Not sure')

    # make no_emplyees category ordered, and integer while we're at it
    def min_emps_map(x):
        if x == 'More than 1000':
            return(1000)
        else:
            pos = x.find('-')
            return(int(x[:pos]))
    mhtech['no_employees'] = mhtech['no_employees'].apply(min_emps_map)

    # impute a value for self-employed depending on number of employees
    mhtech.loc[mhtech['no_employees'] == 1 & mhtech['self_employed'].isnull(), 'self_employed'] = 'Yes'
    mhtech.loc[mhtech['no_employees'] > 1 & mhtech['self_employed'].isnull(), 'self_employed'] = 'No'

    # cleaning gender data for analysis
    # no disrespect, but we have to reduce the number of categories to make a meaningful analysis
    # a bigger dataset with greater representation would be a different story
    def categoriseGender(Gender):
        gender=str(Gender).lower().replace('(cis)', '').replace('cis', '').strip()
        if re.search('not sure|-|/|\?|trans|ish', gender):
            return('others')
        elif re.search('fema', gender) or gender == 'f':
            return('female')
        elif re.search('ma', gender) or gender == 'm' or gender == 'msle':
            return('male')
        else:
            return('others')

    mhtech['gender'] = mhtech['gender'].apply(categoriseGender)

    def likertMap(x):
        if x == 'Very difficult' or x == 'Often':
            return(0)
        elif x == 'Somewhat difficult' or x == 'Sometimes':
            return(1)
        elif x == 'Not sure':
            return(2)
        elif x == 'Somewhat easy' or x == 'Rarely':
            return(3)
        elif x == 'Very easy' or x == 'Never':
            return(4)

    def yesnoMap(x):
        if x == 'Yes':
            return(2)
        elif x == 'Not sure':
            return(1)
        elif x == 'No':
            return(0)


    def regionMap(x):
        if x in ['United States', 'Canada']:
            return('USCA')
        elif x in ['United Kingdom', 'France', 'Netherlands', 'Switzerland',
                  'Germany', 'Austria', 'Ireland', 'Belgium',
                  'Sweden', 'Finland', 'Norway', 'Denmark',
                  'Italy', 'Spain', 'Portugal', 'Slovenia',
                  'Greece', 'Bosnia and Herzegovina', 'Croatia',
                  'Bulgaria', 'Poland', 'Russia', 'Latvia', 'Romania',
                  'Hungary', 'Moldova', 'Georgia', 'Czech Republic']:
            return('EUR')
        elif x in ['Mexico', 'Brazil', 'Costa Rica', 'Colombia', 'Uruguay', 'Bahamas']:
            return('CSA')
        elif x in ['Nigeria', 'South Africa', 'Zimbabwe', 'Israel']:
            return('MEAF')
        elif x in ['India', 'China', 'Philippines', 'Thailand', 'Japan',
                  'Singapore', 'Australia', 'New Zealand']:
            return('APAC')
        else:
            return(x)

    mhtech['region'] = mhtech.country.apply(regionMap)
    mhtech.drop('country', 1, inplace=True)
    mhtech.leave = mhtech.leave.apply(likertMap)
    # this will also fix the missing vals
    mhtech.work_interfere = mhtech.work_interfere.apply(likertMap)

    # code all the binary indep vars
    for c in mhtech.columns:
        if c in ['age', 'gender', 'country', 'leave', 'work_interfere', 'comments', 'no_employees', 'region', 'treatment']:
            continue
        else:
            mhtech[c] = mhtech[c].apply(yesnoMap)

    # should be no more missing data
    # print('missing values')
    # print(mhtech.isnull().sum())
    # print()

    var_names = mhtech.columns.to_list()
    var_names = [vn for vn in var_names if vn != 'treatment']
    var_names.append('treatment')
    mhtech = mhtech[var_names] # put the class col at the end

    mhtech.to_csv('CHIRPS\\datafiles\\mhtech14.csv.gz', index=False, compression='gzip')
    '''

if True:
    '''
    file = '2016.csv'
    archive = zipfile.ZipFile('CHIRPS/source_datafiles/mhtech.zip', 'r')
    lines = archive.read(file).decode("utf-8").split('\r\n')
    archive.close()

    # convert strings to var lists - deal with some free text issues
    lines = [lines[i].replace("<strong>", "").replace("</strong>", "").
             replace("Yes - they all did", "all").replace("No - none did", "none").
             replace("Some did", "some").replace("Yes - always", "always").
             replace("Overall - how", "how").replace("None did", "none").
             replace("I don't know", "not sure").replace("No - I don't think they would", "no").
             replace("No - I don't think it would", "no").replace("Yes - they do", "definitely").
             replace("Yes - I think it would", "yes").replace("Yes - I know several", "several").
             replace("Yes - it has", "definitely").replace("No - they do not", "definitely not").
             replace("Yes - I know several", "several").replace("I know some", "some").
             replace("No - I don't know any", "none").replace("Not applicable to me", "not applicable").
             replace("No - at none of my previous employers", "no").replace("Some of my previous employers", "some").
             replace("Sometimes - if it comes up", "sometimes").replace("No - because it doesn't matter", "no need").
             replace("No - because it would impact me negatively", "no").replace("Yes - I was aware of all of them", "all").
             replace("I was aware of some", "some").replace("N/A (not currently aware)", "no").
             replace("No - I only became aware later", "no").replace("Yes - I experienced", "yes").
             replace("Maybe/Not sure", "not sure").replace("Yes - I observed", "yes").
             replace("Yes - at all of my previous employers", "yes").replace("performance - ", "yes").
             replace("Honestly,I", "I").replace("with ,y", "with my").replace("pay,ents", "payments").
             replace("awareness,more", "awareness - more").replace("crazy,", "crazy").replace("neutral,", "neutral").
             replace("talk,", "talk").replace("1,2,3", "1-2-3").replace("A,B,C", "A-B-C").
             replace(" ,eating", " - eating").replace("grief,coping", "grief - coping").
             replace("Not eligible for coverage / N/A", "not eligible").#replace("N/A", "not eligible").
             replace("I am not sure", "Not sure").replace("Neither easy nor difficult", "neutral").
             replace("No - Not sure any", "none").replace("I'm not sure", "not sure").replace("No - it has not", "no").
             replace("Some of them", "some").replace("Yes - all of them", "all").replace("None of them", "none").
             replace("Yes - I think they would", "yes").replace("Maybe", "possibly").
             replace("Somewhat open", "some").replace("Somewhat not open", "less").replace("Very open", "always").
             replace("Not open at all", "none").replace("Neutral", "neutral").replace("Yes", "yes").replace("No", "no").
             replace('Somewhat ', '').replace('Very', 'very').replace('Unsure', 'not sure').replace('Sometimes', 'sometimes').
             split(',') for i in range(len(lines))]
    names = [nm.replace('"', '').strip() for nm in lines[0]]
    lines = lines[1:]

    # np.array(names)[[("disorder" in n) for n in names]]
    lines.pop() # empty final row
    mhtech = pd.DataFrame(lines, columns=names)

    # make no_emplyees category ordered, and integer while we're at it
    def minempsMap(x):
        if x == 'More than 1000':
            return(1000)
        elif x == 'Jun-25':
            return(6)
        elif x == '':
            return(np.nan)
        else:
            pos = x.find('-')
            return(int(x[:pos]))
    mhtech['How many employees does your company or organization have?'] = \
        mhtech['How many employees does your company or organization have?'].apply(minempsMap)

    def intyesnoMap(x):
        if str(x) == '1':
            return('yes')
        elif str(x) == '0':
            return('no')
        else:
            return('not sure')

    def notsureMap(x):
        if x == '' or x == 'Maybe' or x == 'N/A' or x == 'no - not sure any':
            return('not sure')
    #     elif x == 'possibly':
    #         return('some')
        else:
            return(x.lower())

    def notapplicMap(x):
        if x == '' or x == 'not applicable (I do not have a mental illness)' or x == 'not eligible' or x == 'N/A':
            return('not applicable')
        else:
            return(x.lower())

    def definitelyyesnoMap(x):
        if x == 'definitely':
            return('yes')
        elif x == 'definitely not':
            return('no')
        else:
            return(x.lower())

    def regionMap(x):
        if x in ['United States', 'Canada', 'Other', 'United States of America']:
            return('USCA')
        elif x in ['United Kingdom', 'France', 'Netherlands', 'Switzerland',
                  'Germany', 'Austria', 'Ireland', 'Belgium',
                  'Sweden', 'Finland', 'Norway', 'norway', 'Denmark',
                  'Italy', 'Spain', 'Portugal', 'Slovenia',
                  'Greece', 'Bosnia and Herzegovina', 'Croatia',
                  'Bulgaria', 'Poland', 'Russia', 'Latvia', 'Romania',
                  'Hungary', 'Moldova', 'Georgia', 'Czech Republic',
                  'Lithuania', 'Estonia', 'Slovakia', 'Serbia']:
            return('EUR')
        elif x in ['Mexico', 'Brazil', 'Costa Rica', 'Colombia', 'Uruguay',
                   'Bahamas', 'Chile', 'Venezuela', 'Argentina', 'Guatemala', 'Ecuador']:
            return('CSA')
        elif x in ['India', 'Nigeria', 'South Africa', 'Zimbabwe', 'Israel',
                   'Pakistan', 'Afghanistan', 'Iran', 'Algeria', 'Bangladesh',
                  'Turkey', 'United Arab Emirates']:
            return('MEAF')
        elif x in ['China', 'Philippines', 'Thailand', 'Japan',
                  'Singapore', 'Australia', 'New Zealand', 'Taiwan',
                   'Brunei', 'Vietnam']:
            return('APAC')
        else:
            return(x)

    # cleaning gender data for analysis
    # no disrespect, but we have to reduce the number of categories to make a meaningful analysis
    # a bigger dataset with greater representation would be a different story

    def genderMap(Gender):
        gender=str(Gender).lower().replace('(cis)', '').replace('cis', '').strip()
        if gender=='' or gender == 'human' or re.search('not sure|-|/|\?|trans|ish|nb|fluid|queer', gender):
            return('others')
        elif re.search('fema', gender) or gender == 'f' or gender == 'fem' or gender == 'woman':
            return('female')
        elif re.search('ma', gender) or gender == 'm' or gender == 'msle' or gender == 'm|' or gender == 'dude':
            return('male')
        else:
            return('others')

    mhtech['What is your gender?'] = mhtech['What is your gender?'].apply(genderMap)

    mhtech['What region do you live in?'] = mhtech['What country do you live in?'].apply(regionMap)
    mhtech.drop('What country do you live in?', 1, inplace=True)
    mhtech['What region do you work in?'] = mhtech['What country do you work in?'].apply(regionMap)
    mhtech.drop('What country do you work in?', 1, inplace=True)


    for c in ['Are you self-employed?',
              'Is your employer primarily a tech company/organization?',
              'Is your primary role within your company related to tech/IT?',
              'Do you have medical coverage (private insurance or state-provided) which includes treatment of \xa0mental health issues?',
              'Do you have previous employers?',
              'Have you ever sought treatment for a mental health issue from a mental health professional?'
             ]:
        mhtech[c] = mhtech[c].apply(intyesnoMap)

    for c in ['Does your employer provide mental health benefits as part of healthcare coverage?',
             'Do you know the options for mental health care available under your employer-provided coverage?',
             'Has your employer ever formally discussed mental health (for example - as part of a wellness campaign or other official communication)?',
             'Does your employer offer resources to learn more about mental health concerns and options for seeking help?',
             'Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?',
             'Do you think that discussing a mental health disorder with your employer would have negative consequences?',
             'Do you think that discussing a physical health issue with your employer would have negative consequences?',
             'Would you feel comfortable discussing a mental health disorder with your coworkers?',
             'Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?',
             'Do you feel that your employer takes mental health as seriously as physical health?',
             'Have you heard of or observed negative consequences for co-workers who have been open about mental health issues in your workplace?',
             'Do you know local or online resources to seek help for a mental health disorder?',
             'Have your previous employers provided mental health benefits?',
             'Were you aware of the options for mental health care provided by your previous employers?',
             'Did your previous employers ever formally discuss mental health (as part of a wellness campaign or other official communication)?',
             'Did your previous employers provide resources to learn more about mental health issues and how to seek help?',
             'Do you think that discussing a mental health disorder with previous employers would have negative consequences?',
             'Do you think that discussing a physical health issue with previous employers would have negative consequences?',
             'Would you have been willing to discuss a mental health issue with your previous co-workers?',
             'Would you have been willing to discuss a mental health issue with your direct supervisor(s)?',
             'Did you feel that your previous employers took mental health as seriously as physical health?',
             'Did you hear of or observe negative consequences for co-workers with mental health issues in your previous workplaces?',
             'If a mental health issue prompted you to request a medical leave from work - asking for that leave would be:',
             'Have you observed or experienced an unsupportive or badly handled response to a mental health issue in your current or previous workplace?',
             'Have your observations of how another individual who discussed a mental health disorder made you less likely to reveal a mental health issue yourself in your current workplace?']:
        mhtech[c] = mhtech[c].apply(notsureMap)

    for c in ['If you have been diagnosed or treated for a mental health disorder - do you ever reveal this to clients or business contacts?',
             'If you have revealed a mental health issue to a client or business contact - do you believe this has impacted you negatively?',
             'If you have been diagnosed or treated for a mental health disorder - do you ever reveal this to coworkers or employees?',
             'If you have revealed a mental health issue to a coworker or employee - do you believe this has impacted you negatively?',
             'Do you believe your productivity is ever affected by a mental health issue?',
             'If yes - what percentage of your work time (time performing primary or secondary job functions) is affected by a mental health issue?',
             'Was your anonymity protected if you chose to take advantage of mental health or substance abuse treatment resources with previous employers?',
             'How willing would you be to share with friends and family that you have a mental illness?',
             'Have your observations of how another individual who discussed a mental health disorder made you less likely to reveal a mental health issue yourself in your current workplace?',
             'If you have a mental health issue - do you feel that it interferes with your work when being treated effectively?',
             'If you have a mental health issue - do you feel that it interferes with your work when NOT being treated effectively?',
             'Do you work remotely?']:
        mhtech[c] = mhtech[c].apply(notapplicMap)

    for c in ['Do you think that team members/co-workers would view you more negatively if they knew you suffered from a mental health issue?',
             'Do you feel that being identified as a person with a mental health issue would hurt your career?']:
        mhtech[c] = mhtech[c].apply(definitelyyesnoMap)

    mhtech.iloc[:,1].fillna(0, inplace=True)
    mhtech.iloc[:,26].replace('no', 'none', inplace=True)
    mhtech.iloc[:,29].replace('always', 'yes', inplace=True)
    for c in range(30, 36):
        mhtech.iloc[:,c] = mhtech.iloc[:,c].replace('none', 'no').replace('all', 'yes').replace('some', 'possibly')
    for c in range(57, 59):
        mhtech.iloc[:,c] = mhtech.iloc[:,c].replace('', 'non-US')

    mhtech['mh1'] = mhtech['Have you ever sought treatment for a mental health issue from a mental health professional?']
    mhtech['mh2'] = mhtech['Have you been diagnosed with a mental health condition by a medical professional?']
    mhtech['mh3'] = mhtech['Do you currently have a mental health disorder?']

    mhtech.to_csv('CHIRPS\\datafiles\\mhtech16.csv.gz', index=False, compression='gzip')
    '''
# incomplete
# def make_mhtech1718(file):
#     archive = zipfile.ZipFile('CHIRPS/source_datafiles/mhtech.zip', 'r')
#     lines = archive.read(file).replace(b"\xa0", b" ").decode("utf-8", 'ignore').split('\r\n')
#     archive.close()
#     # convert strings to var lists - deal with some free text issues
#     lines = [lines[i].replace("<strong>", "").replace("</strong>", "").
#              replace("Yes - they all did", "all").replace("No - none did", "none").
#              replace("Some did", "some").replace("Yes - always", "Always").
#              replace("Overall - how", "How").
#              replace("I don't know", "Not sure").replace("No - I don't think they would", "No").
#              replace("No - I don't think it would", "No").replace("Yes - they do", "Definitely").
#              replace("Yes - I think it would", "Yes").replace("Yes - I know several", "Several").
#              replace("Yes - it has", "Definitely").replace("No - they do not", "Definitely not").
#              replace("Yes - I know several", "Several").replace("I know some", "Some").
#              replace("No - I don't know any", "None").replace("Not applicable to me", "N/A").
#              replace("No - at none of my previous employers", "No").replace("Some of my previous employers", "Some").
#              replace("Sometimes - if it comes up", "Sometimes").replace("No - because it doesn't matter", "No need").
#              replace("No - because it would impact me negatively", "No").replace("Yes - I was aware of all of them", "All").
#              replace("I was aware of some", "Some").replace("N/A (not currently aware)", "No").
#              replace("No - I only became aware later", "No").replace("Yes - I experienced", "Yes").
#              replace("Maybe/Not sure", "Not sure").replace("Yes - I observed", "Obs").
#              replace("Yes - at all of my previous employers", "Yes").replace("performance - ", "Yes").
#              replace("Honestly,I", "I").replace("with ,y", "with my").replace("pay,ents", "payments").
#              replace("awareness,more", "awareness - more").replace("crazy,", "crazy").replace("neutral,", "neutral").
#              replace("talk,", "talk").replace("1,2,3", "1-2-3").replace("A,B,C", "A-B-C").
#              replace(" ,eating", " - eating").replace("grief,coping", "grief - coping").
#              split(',') for i in range(len(lines))]
#     names = [nm.replace('"', '').strip() for nm in lines[0]]
#     lines = lines[1:]
#
#     lines.pop() # empty final row
#     return(pd.DataFrame(lines, columns=names))
#
# # make no_emplyees category ordered, and integer while we're at it
# def min_emps_map(x):
#     if x == 'More than 1000':
#         return(1000)
#     else:
#         pos = x.find('-')
#         return(int(x[:pos]))
# mhtech['no_employees'] = mhtech['no_employees'].apply(min_emps_map)
#
#
# mhtech1718 = pd.concat([make_mhtech1718(file = '2017.csv'), make_mhtech1718(file = '2018.csv')])

if True:
    '''
    file = 'yps.csv'
    archive = zipfile.ZipFile('CHIRPS/source_datafiles/yps.zip', 'r')
    lines = archive.read(file).decode("utf-8").split('\n')
    archive.close()

    lines = [lines[i].replace(', ', ' - ').replace('"', '').
             replace(' smoker', '').replace(' smoked', '').
             replace(' smoking', '').replace(' drinker', '').
             replace('drink ', '').replace('i am always ', '').
             replace('i am often ', '').replace('running ', '').
             replace('few hours a day', 'sometimes').replace('less than one hour a day', '<1 hours').
             replace('most of the day', 'many hours').replace('no time at all', 'never').
             replace('female', 'f').replace('male', 'm').replace(' handed', '').
             replace('currently a primary school pupil', 'primary').
             replace(' school', '').replace(' degree', '').replace('/bachelor', '').
             replace('block of ', '').replace('/bungalow', '').
             split(',') for i in range(len(lines))]
    names = [nm for nm in lines[0]]
    lines = lines[1:]
    lines.pop() # empty final row
    yps = pd.DataFrame(lines, columns=names)

    def numericMap(x):
        if x == '':
            return(np.nan)
        else:
            return(np.float16(x))

    def stringMap(x):
        if x == '':
            return(np.nan)
        else:
            return(x)

    cat_vars = ['Smoking', 'Alcohol',
                'Punctuality', 'Lying',
                'Internet usage', 'Gender',
                'Left - right', 'Education', 'Only child',
                'Village - town', 'House - flats']

    impnum = SimpleImputer(missing_values=np.nan, strategy='median')
    impcat = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    for c in yps.columns:
        if c not in cat_vars:
            yps[c] = yps[c].apply(numericMap)
            impnum.fit(np.array(yps[c]).reshape(-1, 1))
            yps[c] = impnum.transform(np.array(yps[c]).reshape(-1, 1))
        else:
            yps[c] = yps[c].apply(stringMap)
            impcat.fit(np.array(yps[c]).reshape(-1, 1))
            yps[c] = impcat.transform(np.array(yps[c]).reshape(-1, 1))

    yps.to_csv('CHIRPS\\datafiles\\yps.csv.gz', index=False, compression='gzip')
    '''

if True:
    '''
    random_state = 123
    file = 'noshow.csv'
    archive = zipfile.ZipFile('CHIRPS/source_datafiles/noshow.zip', 'r')
    lines = archive.read(file).decode("utf-8").split('\r\n')
    archive.close()

    lines = [lines[i].replace(', ', ' - ').replace('"', '').
             replace('No-show', 'no_show').split(',') for i in range(len(lines))]
    names = [nm for nm in lines[0]]
    lines = lines[1:]
    lines.pop() # empty final row
    noshow = pd.DataFrame(lines, columns=names)

    def unicodeMap(x):
        return(unicodedata.normalize('NFKD', x).encode('ascii','ignore').decode())
    noshow['Neighbourhood'] = noshow['Neighbourhood'].apply(unicodeMap)
    noshow['SchedDay'] = pd.to_datetime(noshow.ScheduledDay).dt.day_name()
    noshow['SchedMonth'] = pd.to_datetime(noshow.ScheduledDay).dt.month_name()
    noshow['ApptDay'] = pd.to_datetime(noshow.AppointmentDay).dt.day_name()
    noshow['ApptMonth'] = pd.to_datetime(noshow.AppointmentDay).dt.month_name()
    # get a date difference between booking and appointment
    noshow['LagDays'] = pd.to_datetime(noshow.AppointmentDay) - pd.to_datetime(noshow.ScheduledDay)
    noshow.LagDays.loc[(pd.to_datetime(noshow.AppointmentDay) - pd.to_datetime(noshow.ScheduledDay)) < \
                       (pd.to_datetime(1) - pd.to_datetime(0))] = pd.to_datetime(1) - pd.to_datetime(1)
    noshow.LagDays = noshow.LagDays / pd.to_timedelta(1, unit='D') # convert to float of days
    noshow.drop(columns=['PatientId', 'AppointmentID' , 'ScheduledDay', 'AppointmentDay'], inplace=True)
    var_names = noshow.columns.to_list()
    var_names = [vn for vn in var_names if vn != 'no_show']
    var_names.append('no_show')
    noshow = noshow[var_names] # put the class col at the end
    noshow.to_csv('CHIRPS\\datafiles\\noshow.csv.gz', index=False, compression='gzip')
    samp = noshow.sample(frac=0.2, random_state=random_state)
    samp.reset_index(drop=True, inplace=True)
    samp.to_csv('CHIRPS\\datafiles\\noshow_samp.csv.gz', index=False, compression='gzip')
    small_samp = noshow.sample(frac=0.02, random_state=random_state)
    small_samp.reset_index(drop=True, inplace=True)
    small_samp.to_csv('CHIRPS\\datafiles\\noshow_small_samp.csv.gz', index=False, compression='gzip')
    '''

if True:
    '''
    file = 'wdbc.data'
    archive = zipfile.ZipFile('CHIRPS/source_datafiles/breastwisc.zip', 'r')
    lines = archive.read(file).decode("utf-8").split('\n')
    archive.close()

    lines = [lines[i].split(',') for i in range(len(lines))]
    lines.pop()
    names = ['id', 'mb'] + ['f_' + str(i + 1) for i in range(30)] # [nm for nm in lines[0]]
    breast = pd.DataFrame(lines, columns=names)
    breast.drop(columns='id', inplace=True)
    var_names = [vn for vn in breast.columns if vn != 'mb']
    var_names.append('mb')
    breast = breast[var_names] # put the class col at the end

    breast.to_csv('CHIRPS\\datafiles\\breast.csv.gz', index=False, compression='gzip')
    '''

if True:
    '''
    file = 'agaricus-lepiota.data'
    archive = zipfile.ZipFile('CHIRPS/source_datafiles/agaricus-lepiota.zip', 'r')
    lines = archive.read(file).decode("utf-8").split('\n')
    archive.close()

    lines = [lines[i].split(',') for i in range(len(lines))]
    lines.pop()
    names = ['edible', 'cshape', 'csurface', 'ccolor', 'bruises', 'odor',
            'gattach', 'gspace', 'gsize', 'gcolor', 'sshape', 'sroot',
            'ssurfaring', 'ssurfbring', 'scoloraring', 'scolorbring',
            'vtype', 'vcolor', 'rnum', 'rtype', 'sporecolor', 'pop', 'hab']
    mush = pd.DataFrame(lines, columns=names)

    var_names = [vn for vn in mush.columns if vn != 'edible']
    var_names.append('edible')
    mush = mush[var_names] # put the class col at the end
    mush.to_csv('CHIRPS\\datafiles\\mush.csv.gz', index=False, compression='gzip')
    '''

if True:
    '''
    file = 'risk_factors_cervical_cancer.csv'
    archive = zipfile.ZipFile('CHIRPS/source_datafiles/cervical.zip', 'r')
    lines = archive.read(file).decode("utf-8").split('\n')
    archive.close()

    lines = [lines[i].split(',') for i in range(len(lines))]
    names = lines[0]
    lines = lines[1:]
    lines.pop()

    cervical = pd.DataFrame(lines, columns=names)
    cervical.drop(columns=['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis'], inplace=True) # too many missing

    var_types = ['continuous',
                'continuous',
                'continuous',
                'continuous',
                'nominal',
                'continuous',
                'continuous',
                'nominal',
                'continuous',
                'nominal',
                'continuous',
                'nominal',
                'continuous',
                'nominal',
                'nominal',
                'nominal',
                'nominal',
                'nominal',
                'nominal',
                'nominal',
                'nominal',
                'nominal',
                'nominal',
                'nominal',
                'nominal',
                'continuous',
                'nominal',
                'nominal',
                'nominal',
                'nominal',
                'nominal',
                'nominal',
                'nominal',
                'nominal']

    def numericMap(x):
        if x == '?':
            return(np.nan)
        else:
            return(np.float16(x))

    def stringMap(x):
        if x in ['1', '1.0', '1.']:
            return('1')
        elif x in ['0', '0.0', '0.']:
            return('0')
        else:
            return(x)

    def truefalseMap(x):
        if x == '1':
            return('T')
        else:
            return('F')

    # some numbers missings are as much as 14%
    impnum = SimpleImputer(missing_values=np.nan, strategy='median')
    impcat = SimpleImputer(missing_values='?', strategy='most_frequent')
    for c, vt in zip(cervical.columns, var_types):
        if vt == 'continuous':
            cervical[c] = cervical[c].apply(numericMap)
            impnum.fit(np.array(cervical[c]).reshape(-1, 1))
            cervical[c] = impnum.transform(np.array(cervical[c]).reshape(-1, 1))
        else:
            cervical[c] = cervical[c].apply(stringMap)
            impcat.fit(np.array(cervical[c]).reshape(-1, 1))
            cervical[c] = impcat.transform(np.array(cervical[c]).reshape(-1, 1))

    for c in ['Hinselmann', 'Schiller', 'Citology', 'Biopsy']:
        cervical[c] = cervical[c].apply(truefalseMap)

    cervical.to_csv('CHIRPS\\datafiles\\cervical.csv.gz', index=False, compression='gzip')
    '''

if True:
    '''
    random_state = 123
    file = 'thyroid0387.data'
    archive = zipfile.ZipFile('CHIRPS/source_datafiles/thyroid0387.zip', 'r')
    lines = archive.read(file).decode("utf-8").split('\n')
    archive.close()
    lines = [lines[i].split(',') for i in range(len(lines))]
    lines.pop()
    names = ['age', 'sex', 'on thyroxine', 'query on thyroxine',
             'on antithyroid medication', 'sick', 'pregnant',
             'thyroid surgery', 'I131 treatment', 'query hypothyroid',
             'query hyperthyroid', 'lithium', 'goitre', 'tumor',
             'hypopituitary', 'psych', 'TSH measured', 'TSH',
             'T3 measured', 'T3', 'TT4 measured', 'TT4', 'T4U measured',
             'T4U', 'FTI measured', 'FTI', 'TBG measured', 'TBG', 'referral source',
            'diagnosis']
    # last item is diagnosis and id.
    # diagnosis is - or a char string.
    # id is a number surrounded by []
    thyroid = pd.DataFrame(lines, columns=names)
    def diagnosisMap(x):
        diag = re.sub(r'\[[0-9]*\]', '', x)
        if diag == '-':
            return('Normal')
        else:
            return('Abnormal')

    # some crazy ages
    def ageMap(x):
        x = np.float16(x)
        if x == np.inf or x > 150:
            return(np.nan)
        else:
            return(x)

    def numericMap(x):
        if x == '?' or x == '':
            return(np.nan)
        else:
            return(np.float16(x))

    def truefalseMap(x):
        if x == 'f':
            return(0)
        else:
            return(1)

    impnum = SimpleImputer(missing_values=np.nan, strategy='median')
    impzero = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0.0)

    thyroid['diagnosis'] = thyroid.diagnosis.apply(diagnosisMap)
    thyroid['age'] = thyroid.age.apply(ageMap)
    impnum.fit(np.array(thyroid['age']).reshape(-1, 1))
    thyroid['age'] = impnum.transform(np.array(thyroid['age']).reshape(-1, 1))
    for c in ['TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']:
        thyroid[c] = thyroid[c].apply(numericMap)
        impzero.fit(np.array(thyroid[c]).reshape(-1, 1))
        thyroid[c] = impzero.transform(np.array(thyroid[c]).reshape(-1, 1))

    for c in thyroid.columns:
        if c not in ['sex', 'age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG', 'referral source', 'diagnosis']:
            thyroid[c] = thyroid[c].apply(truefalseMap)

    thyroid.to_csv('CHIRPS\\datafiles\\thyroid.csv.gz', index=False, compression='gzip')

    samp = thyroid.sample(frac=0.25, random_state=random_state)
    samp.reset_index(drop=True, inplace=True)
    samp.to_csv('CHIRPS\\datafiles\\thyroid_samp.csv.gz', index=False, compression='gzip')
    '''

if True:
    '''
    data = pd.read_csv('C:\\Users\\id126493\\Documents\\GitHub\\explain_te\\CHIRPS\\source_datafiles\\heart.csv')
    data.to_csv('CHIRPS\\datafiles\\heart.csv.gz', index=False, compression='gzip')

    data = pd.read_csv('C:\\Users\\id126493\\Documents\\GitHub\\explain_te\\CHIRPS\\datafiles\\heart.csv.gz',
                        compression='gzip')
    '''

if True:
    '''
    import pandas as pd
    data = pd.read_csv('C:\\Users\\id126493\\Documents\\GitHub\\explain_te\\CHIRPS\\source_datafiles\\diabetic_retinopathy.csv', header=None)
    data.rename(columns={0 : 'qa', 1 : 'ps', 2 : 'ma0.5', 3 : 'ma0.6',
                         4 : 'ma0.7', 5 : 'ma0.8', 6 : 'ma0.9', 7 : 'ma1.0',
                         8 : 'ex0.5', 9 : 'ex0.6', 10 : 'ex0.7', 11 : 'ex0.8',
                         12 : 'ex0.9', 13 : 'ex1.0', 14 : 'exm1', 15 : 'exm2',
                         16 : 'eucmac', 17 : 'diaopt', 18 : 'amfm', 19 : 'dr'}, inplace=True)
    dr = ['yes' if ddr == 1 else 'no' for ddr in data.dr]
    data['dr'] = dr

    # 0) The binary result of quality assessment. 0 = bad quality 1 = sufficient quality.
    # 1) The binary result of pre-screening, where 1 indicates severe retinal abnormality and 0 its lack.
    # 2-7) The results of MA detection. Each feature value stand for the
    # number of MAs found at the confidence levels alpha = 0.5, . . . , 1, respectively.
    # 8-15) contain the same information as 2-7) for exudates. However,
    # as exudates are represented by a set of points rather than the number of
    # pixels constructing the lesions, these features are normalized by dividing the
    # number of lesions with the diameter of the ROI to compensate different image
    # sizes.
    # Note - 2-7 and 8-15 are not equal in number. Not sure what to call the last two ex.
    # 16) The euclidean distance of the center of
    # the macula and the center of the optic disc to provide important information
    # regarding the patientâ€™s condition. This feature
    # is also normalized with the diameter of the ROI.
    # 17) The diameter of the optic disc.
    # 18) The binary result of the AM/FM-based classification.
    # 19) Class label. 1 = contains signs of DR (Accumulative label for the Messidor classes 1, 2, 3), 0 = no signs of DR.


    data.to_csv('CHIRPS\\datafiles\\diaretino.csv.gz', index=False, compression='gzip')
    '''

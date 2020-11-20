import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import numpy

class Load_Dataset:
    """Class for loading preprocessed datasets"""

    def rul_finder(mapper):
        RULs = []
        RULsInd = []
        cnt = 1
        for m in mapper:
            RULsInd.append(cnt)
            cnt += 1
            RULs.append(m[1])
        return RULsInd, RULs

    def load_data_turbofan(plot_RULs=False):
        feature_names = ['u', 't', 'os_1', 'os_2', 'os_3'] #u:unit, t:time, s:sensor
        feature_names += ['s_{0:02d}'.format(s + 1) for s in range(26)]
        fd = {}
        for i in range(4):
            p = 'datasets/CMAPSSData/train_FD00'+ str(i+1) +'.txt'
            df_train = pd.read_csv(p, sep= ' ', header=None, names=feature_names, index_col=False)
            mapper = {}
            for unit_nr in df_train['u'].unique():
                mapper[unit_nr] = df_train['t'].loc[df_train['u'] == unit_nr].max()#max time einai to rul tou
            # calculate RUL = time.max() - time_now for each unit
            df_train['RUL'] = df_train['u'].apply(lambda nr: mapper[nr]) - df_train['t']

            p = 'datasets/CMAPSSData/test_FD00'+ str(i+1) +'.txt'
            df_test = pd.read_csv(p, sep= ' ', header=None, names=feature_names, index_col=False)
            p = 'datasets/CMAPSSData/RUL_FD00'+ str(i+1) +'.txt'
            df_RUL = pd.read_csv(p, sep= ' ', header=None, names=['RUL_actual'], index_col=False)
            temp_mapper = {}
            for unit_nr in df_test['u'].unique():
                temp_mapper[unit_nr] = df_test['t'].loc[df_test['u'] == unit_nr].max()#max time einai to rul tou

            mapper_test = {}
            cnt = 1
            for mt in df_RUL.values:
                mapper_test[cnt]=mt[0]+temp_mapper[cnt]
                cnt += 1
            df_test['RUL'] = df_test['u'].apply(lambda nr: mapper_test[nr]) - df_test['t']

            if plot_RULs:
                mapper = sorted(mapper.items(), key=lambda kv: kv[1])
                plt.figure(figsize=(10, 5))
                ax1 = plt.subplot(121)
                ax1.margins(0.05)           # Default margin is 0.05, value 0 means fit
                RULsInd, RULs = rul_finder(mapper)
                ax1.plot(RULsInd, RULs)
                ax1.set_title('Fault Mode '+ str(i+1)+': Train set')
                ax1.set_xlabel('Unit_id')
                ax1.set_ylabel('RUL')  
                mapper_test = sorted(mapper_test.items(), key=lambda kv: kv[1])
                ax2 = plt.subplot(122)
                ax2.margins(0.05)           # Default margin is 0.05, value 0 means fit
                tRULsInd, tRULs = rul_finder(mapper_test)
                ax2.plot(tRULsInd,tRULs)
                ax2.set_title('Fault Mode '+ str(i+1)+': Test set')
                ax2.set_xlabel('Unit_id')
                ax2.set_ylabel('RUL')
                plt.show()
                print('[FaultMode'+ str(i+1) +']','Train Min:',RULs[0],' Max:',RULs[-1],'| Test Min:',tRULs[0],' Max',tRULs[-1])

            s = 'FaultMode'+ str(i+1) +''
            fd[s] = {'df_train': df_train, 'df_test': df_test}
        feature_names.append('RUL')
        return fd, feature_names


    def load_hate_data(preprocessed = True, stem = True):
        data = pd.read_csv("datasets/hateSpeechDataV2.csv", delimiter=';')

        XT = data['comment'].values
        X = []
        yT = data['isHate'].values #Add all the labels here :/
        y = []
        for yt in yT:
            if yt>=0.5:
                y.append(int(1))
            else:
                y.append(int(0))
        for x in XT:
            if preprocessed:
                X.append(Load_Dataset.my_clean1(text=str(x), stops=False, stemming=stem))
            else:
                X.append(x)
        return numpy.array(X),numpy.array(y)

    def load_hate_unsupervised_data(preprocessed = True, stem = True):
        missing_values = ["?"]
        df = pd.read_csv("datasets/hate_tweets.csv",na_values = missing_values)
        X = df['tweet'].values
        y = df['class'].values
        y = [1 if i==0 else 0 for i in y]
        x = []
        if preprocessed:
            for i in X:
                x.append(Load_Dataset.my_clean1(text=str(i), stops=False, stemming=stem))
        else:
            x = X
        return x,y
        
    def load_smsspam(preprocessed=True,stemming=True):
        """load_hate_speech function returns the smsspam dataset
        Args:
            preprocessed: If true it returns the dataset preprocessed.
        Return:
            X: Data Instances
            y: Targets
            class_names: ['spam', 'ham']
        """
        df = pd.read_csv('datasets/spam.csv', encoding='latin-')
        X = df['v2'].values
        y = df['v1'].values
        le = LabelEncoder()
        y = le.fit_transform(y)
        class_names = ['ham', 'spam']
        if preprocessed:
            X = Load_Dataset.pre_processing(X,stemming)
        return X,y,class_names

    def pre_processing(pX, stemming=True):
        clean_tweet_texts = []
        for t in pX:
            clean_tweet_texts.append((Load_Dataset.my_clean2(t, False, stemming, 2)))  # You can add one more clean()
        return clean_tweet_texts

    def my_clean1(text, stops=False, stemming=False):
        text = str(text)
        text = re.sub(r" US ", " american ", text)
        text = text.lower().split()
        text = " ".join(text)
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"don't", "do not ", text)
        text = re.sub(r"aren't", "are not ", text)
        text = re.sub(r"isn't", "is not ", text)
        text = re.sub(r"%", " percent ", text)
        text = re.sub(r"that's", "that is ", text)
        text = re.sub(r"doesn't", "does not ", text)
        text = re.sub(r"he's", "he is ", text)
        text = re.sub(r"she's", "she is ", text)
        text = re.sub(r"it's", "it is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"\s{2,}", " ", text)
        text = text.lower().split()
        text = [w for w in text if len(w) >= 2]
        if stemming and stops:
            text = [word for word in text if word not in stopwords.words('english')]
            wordnet_lemmatizer = WordNetLemmatizer()
            englishStemmer = SnowballStemmer("english", ignore_stopwords=True)
            text = [englishStemmer.stem(word) for word in text]
            text = [wordnet_lemmatizer.lemmatize(word) for word in text]
            # text = [lancaster.stem(word) for word in text]
            text = [word for word in text if word not in stopwords.words('english')]
        elif stops:
            text = [word for word in text if word not in stopwords.words('english')]
        elif stemming:
            wordnet_lemmatizer = WordNetLemmatizer()
            englishStemmer = SnowballStemmer("english", ignore_stopwords=True)
            text = [englishStemmer.stem(word) for word in text]
            text = [wordnet_lemmatizer.lemmatize(word) for word in text]
        text = " ".join(text)
        return text

    def my_clean2(text,stops = False,stemming = False,minLength = 2):
        text = str(text)
        text = re.sub(r" US ", " u s ", text)
        text = text.lower().split()
        if stemming and stops:
            text = [word for word in text if word not in stopwords.words('english')]
            wordnet_lemmatizer = WordNetLemmatizer()
            englishStemmer = SnowballStemmer("english", ignore_stopwords=False)
            text = [englishStemmer.stem(word) for word in text]
            text = [wordnet_lemmatizer.lemmatize(word) for word in text]
            text = [word for word in text if word not in stopwords.words('english')]
        elif stops:
            text = [word for word in text if word not in stopwords.words('english')]
        elif stemming:
            wordnet_lemmatizer = WordNetLemmatizer()
            englishStemmer = SnowballStemmer("english", ignore_stopwords=False)
            text = [englishStemmer.stem(word) for word in text]
            text = [wordnet_lemmatizer.lemmatize(word) for word in text]
        text = " ".join(text)
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"don't", "do not ", text)
        text = re.sub(r"aren't", "are not ", text)
        text = re.sub(r"isn't", "is not ", text)
        text = re.sub(r"%", " percent ", text)
        text = re.sub(r"that's", "that is ", text)
        text = re.sub(r"doesn't", "does not ", text)
        text = re.sub(r"he's", "he is ", text)
        text = re.sub(r"she's", "she is ", text)
        text = re.sub(r"it's", "it is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r" e - mail ", " email ", text)
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ", text)
        text = re.sub(r";", " ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", " ", text)
        text = re.sub(r"\+", " ", text)
        text = re.sub(r"\-", " ", text)
        text = re.sub(r"\=", " ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r" j k ", " jk ", text)
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", text) #Removes every number
        text = text.lower().split()
        text = [w for w in text if len(w) >= minLength]
        if stemming and stops:
            text = [word for word in text if word not in stopwords.words('english')]
            wordnet_lemmatizer = WordNetLemmatizer()
            englishStemmer = SnowballStemmer("english", ignore_stopwords=False)
            text = [englishStemmer.stem(word) for word in text]
            text = [wordnet_lemmatizer.lemmatize(word) for word in text]
            text = [word for word in text if word not in stopwords.words('english')]
        elif stops:
            text = [word for word in text if word not in stopwords.words('english')]
        elif stemming:
            wordnet_lemmatizer = WordNetLemmatizer()
            englishStemmer = SnowballStemmer("english", ignore_stopwords=False)
            text = [englishStemmer.stem(word) for word in text]
            text = [wordnet_lemmatizer.lemmatize(word) for word in text]
        text = " ".join(text)
        return text

import contractions
import string
import re
import joblib
import logg
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import nltk

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import  TfidfVectorizer
nltk.download('stopwords')
del_columns = ['book_title', 'book_desc', 'book_genre', 'book_authors', 'book_format']





class Model_Pipeline:
    def __init__(self,  file):

        #self.log = logg.get_class_log(self)
        self.pipeline = joblib.load(file)

    def pipelineData(self, data):
        return self.pipeline.transform(data)

    def predict(self, data):
        return self.pipeline.predict(data)


def preprocess(data_real, colum_name):
    def remove_punctuation(sentence):
        unnecessary_dict = {}
        for symb in string.punctuation:
            unnecessary_dict[symb] = ' '
        unnecessary_dict['\x96'] = ' '
        unnecessary_dict['\x85'] = ' '
        unnecessary_dict['´'] = ' '
        unnecessary_dict['\x97'] = ' '
        unnecessary_dict['…'] = ' '
        unnecessary_dict['’'] = ' '
        unnecessary_dict['\x91'] = ' '

        s = sentence.replace('<br />', '')
        s = s.translate(s.maketrans(unnecessary_dict))
        return s

    def remove_stopwords(sentence):
        without_sw = []
        stop_words = stopwords.words('english')
        stop_words.remove('not')
        stop_words.remove('no')

        words = sentence.split()
        for word in words:
            if word not in stop_words:
                without_sw.append(word)

        res = ' '.join(without_sw)
        return res

    def lemmatize_text(text):
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(w) for w in text.split()]
        return " ".join(words)

    data = data_real.copy()
    # all to lowercase
    data[colum_name] = data[colum_name].str.lower()
    # change contractions (i've, don't ...) to full forms (i have, do not)
    data[colum_name] = data[colum_name].apply(lambda x: " ".join([contractions.fix(word) for word in str(x).split()]))
    # remove punctuation
    data[colum_name] = data[colum_name].apply(remove_punctuation)
    # remove numbers
    data[colum_name] = data[colum_name].apply(lambda s: re.sub('\d+', ' ', s))
    # remove stopwords
    data[colum_name] = data[colum_name].apply(remove_stopwords)
    # remove single letters
    data[colum_name] = data[colum_name].apply(lambda s: re.sub('\b[a-zA-Z]\b', ' ', s))
    # remove excess spaces
    data[colum_name] = data[colum_name].apply(lambda s: re.sub(' +', ' ', s))
    # lemmatize
    data[colum_name] = data[colum_name].apply(lemmatize_text)
    # tokenize
    data[f'tokenized_{colum_name}'] = data[colum_name].apply(lambda s: s.split())
    # add number of words column
    data[f'word_num_{colum_name}'] = data[f'tokenized_{colum_name}'].str.len()

    return data



del_columns = ['campaign_name']

class CleaningTextData(BaseEstimator, TransformerMixin):
    def __init__(self, del_columns=del_columns):
        self.del_columns = del_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        data_new = X.copy()

        data_new = data_new[data_new["original_spend"] > 0.1]

        data_new["date"] = pd.to_datetime(data_new["date"])
        data_new["start_time"] = pd.to_datetime(data_new["start_time"])

        data_new.fb_id = data_new['fb_id'].apply(lambda x: int(x.lstrip('act_')))

        # cleaning

        data_new = preprocess(data_new, 'campaign_name')

        # data_new = pd.concat([data_new,pd.get_dummies(data_new["country_code"])],axis=1)

        data_new = data_new.drop(self.del_columns, axis=1)

        # to string some columns
        self.columns_str = ['tokenized_campaign_name']

        for name in self.columns_str:
            data_new[name] = data_new[name].apply(lambda x: np.nan if len(x) == 0 or x[0] == 'ok' else ' '.join(x))

        # come back to nan
        columns_back = ['tokenized_campaign_name']

        for name in columns_back:
            data_new[name] = data_new[name].apply(lambda x: np.nan if x == 'nan' else x)

        return data_new


class FillingNaN(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='most_frequent'):
        self.strategy = strategy

    #         self.columns_input = columns_input

    def fit(self, X, y=None):
        print('fit filling na')

        data = X.copy()

        self.imputer = SimpleImputer(strategy=self.strategy)
        self.imputer.fit(data)

        return self

    def transform(self, X):
        data = X.copy()
        data_subset_transformed = self.imputer.transform(data)
        data = pd.DataFrame(data_subset_transformed, columns=data.columns)

        return data


class OneHotCountry(BaseEstimator, TransformerMixin):
    def init(self, remainder='passthrough'):
        pass

    #         self.remainder = remainder

    def fit(self, X, y=None):
        data = X.copy()

        values = data.country_code
        self.label_encoder = LabelEncoder()
        #         self.label_encoder.fit(values)

        integer_encoded = self.label_encoder.fit_transform(values)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

        self.onehot_encoder = OneHotEncoder(sparse=False)
        self.onehot_encoder.fit(integer_encoded)

        return self

    def transform(self, X):
        data = X.copy()

        values = data.country_code

        integer_test = self.label_encoder.transform(values)
        integer_test = integer_test.reshape(len(integer_test), 1)

        onehot_test = pd.DataFrame(self.onehot_encoder.transform(integer_test))

        #         onehot_encoded = pd.DataFrame(self.onehot_encoder.transform(integer_encoded))
        data.drop(['country_code'], axis=1, inplace=True)
        df_result = pd.concat([data, onehot_test], axis=1)

        return df_result


columns_idf = ['tokenized_campaign_name']


class TfIdf(BaseEstimator, TransformerMixin):
    def __init__(self, columns_idf=columns_idf, max_features=10000):
        self.columns_idf = columns_idf
        self.max_features = max_features
        self.model_dic = {}

    def fit(self, X, y=None):
        print('fit tfidf')
        data = X.copy()

        for name in self.columns_idf:
            self.model_dic[name] = TfidfVectorizer(stop_words='english', max_features=self.max_features)
            self.model_dic[name].fit(data[name].values)
        return self

    def transform(self, X, y=None):
        data = X.copy()
        data_new = data.copy()

        for name, model in self.model_dic.items():
            print
            data_transformed = model.transform(data[name].values).toarray()
            # model.get_name()
            # model.vocabulary_
            data_transformed = pd.DataFrame(data_transformed,
                                            columns=[f'{name}_{x}' for x in range(data_transformed.shape[1])])
            data_new = pd.concat([data_new, data_transformed], axis=1)

        data_new = data_new.drop(list(self.model_dic.keys()), axis=1)

        return data_new

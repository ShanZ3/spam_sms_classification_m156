#!/usr/bin/env python
# coding: utf-8

# ### Preparation

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

import spacy

import statsmodels.api as sm

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve

from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report

from sklearn.externals import joblib

import os.path

class SMSBase:
    # Spacy library is loading English dictionary.
    _nlp = spacy.load("en")
    
    def __init__(self, filename, frac=0.8):
        self._filename = filename
        self._features = ['class', 'context']
        
        self._df_raw = pd.read_csv(self._filename, sep='\t', names=self._features)
        self.__format_context()
        
        self.__extract_features()
        
        self._group_by_feature = self._df_raw .groupby('class')
        self._counts_by_features = self._group_by_feature.count().to_dict()['context']
        
        self.__split_test_train(frac)
        
    def __format_context(self):
        self._df_raw['context'] =  self._df_raw['context'].map(lambda text : text.rstrip())
        self._df_raw['context'] =  self._df_raw['context'].map(lambda text : text.replace(',', ' ,') if ',' in text else text)
    
    def __extract_features(self):
        self._df_raw['len']= self._df_raw['context'].map(lambda text : len(text))
        self._df_raw['n_words'] = self._df_raw['context'].map(lambda text : len(text.split(' ')))

        #updating features
        self._features = self._df_raw.columns
    
    def __split_test_train(self, frac):
        self._df_train = self._df_raw.sample(frac=frac)
        self._df_test = self._df_raw.drop(self._df_train.index)
    
    def describe(self):
        print('-' * 20 + 'Extended Dataset (Head)' + '-' * 20)
        display(self._df_raw.head())
        
        print('-' * 20 + 'Extended Dataset (Describe)' + '-' * 20)
        display(self._df_raw.describe())
        
        print('-' * 20 + 'Groupby Class (Describe)' + '-' * 20)
        display(self._group_by_feature.describe())
        
    def create_lemmas(self, c):
        tokens = self._nlp(c)
        return [token.lemma_ for token in tokens]
    
    def create_tokens(self, c):
        tokens = self._nlp(c)
        return [token for token in tokens]
    
    
class Util:
        
    def report_classification(model, df_train, df_test, X_features, y_feature):
        
        classes_train = np.unique(df_train[y_feature].values).tolist()
        classes_test = np.unique(df_test[y_feature].values).tolist()
        
        assert (classes_train == classes_test)
        
        classes = classes_train # The order of class is important!
        
        X_train = df_train[X_features].values.tolist()
        X_test = df_test[X_features].values.tolist()
        
        y_train = df_train[y_feature].values.tolist()
        y_test = df_test[y_feature].values.tolist()
        
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        report_cm(y_train, y_test, y_train_pred, y_test_pred, classes)
        
    def report_cm(y_train, y_test, y_train_pred, y_test_pred, classes):
        figure, axes = plt.subplots(1, 2, figsize=(10,5))

        cm_test = confusion_matrix(y_test, y_test_pred)
        df_cm_test = pd.DataFrame(cm_test, index = classes, columns = classes)
        ax = sns.heatmap(df_cm_test, annot=True, ax = axes[0], square= True)
        ax.set_title('Test CM')

        cm_train = confusion_matrix(y_train, y_train_pred)
        df_cm_train = pd.DataFrame(cm_train, index = classes, columns = classes)
        ax = sns.heatmap(df_cm_train, annot=True, ax = axes[1], square= True)
        ax.set_title('Train CM')

        print('-' * 20 + 'Testing Performance' + '-' * 20)
        print(classification_report(y_test, y_test_pred, target_names = classes))
        print('acc: ', metrics.accuracy_score(y_test, y_test_pred))

        print('-' * 20 + 'Training Performance' + '-' * 20)
        print(classification_report(y_train, y_train_pred, target_names = classes))
        print('acc: ', metrics.accuracy_score(y_train, y_train_pred))
        
    
    def plot_cdf(p, 
             ax, 
             deltax=None, 
             xlog=False, 
             xlim=[0, 1], 
             deltay=0.25, 
             ylog=False, 
             ylim=[0,1], 
             xlabel = 'x'):

        df = pd.DataFrame(p, columns=[xlabel])
        display(df.describe())
        
        ecdf = sm.distributions.ECDF(p)
        x = ecdf.x
        y = ecdf.y
        assert len(x) == len(y)
        if deltax is not None:
            x_ticks = np.arange(xlim[0], xlim[1] + deltax, deltax)
            ax.set_xticks(x_ticks)

        ax.set_xlabel(xlabel)
        ax.set_xlim(xlim[0], xlim[1])
        ax.vlines(np.mean(p), min(y), max(y), color='red', label='mean', linewidth=2)
        ax.vlines(np.median(p), min(y), max(y), color='orange', label='median', linewidth=2)
        ax.vlines(np.mean(p) + 2 * np.std(p), min(y), max(y), color='blue', label='mean + 2 * std', linewidth=2)
        ax.vlines(np.mean(p) + 3 * np.std(p), min(y), max(y), color='green', label='mean + 3 * std', linewidth=2)

        y_ticks = np.arange(ylim[0], ylim[1] + deltay, deltay)
        ax.set_ylabel('CDF')
        ax.set_yticks(y_ticks)
        ax.set_ylim(ylim[0], ylim[1])

        if xlog is True:
            ax.set_xscale('log')

        if ylog is True:
            ax.set_yscale('log')


        ax.grid(which='minor', alpha=0.5)
        ax.grid(which='major', alpha=0.9)

        ax.legend(loc=4)

        sns.set_style('whitegrid')
        sns.regplot(x=x, y=y, fit_reg=False, scatter=True, ax = ax)
    
        
    def plot_class_dist(df, by):
        
        x_features = df.columns.drop(by)
        assert 0 < len(x_features)
        
        x_features = x_features[0]
        dist = df.groupby(by)[x_features].size() / len(df)
        display(dist)        
        sns.barplot(x=dist.index, y=dist.values)
        
    def plot_boxplot(df, by, y, ax):
        ax = sns.boxplot(x=by, y=y, data=df[[by,  y]], ax = ax)
        ax.set_yscale('log')
        
    def dump_pickle(obj,filename):
        joblib.dump(obj, filename)
        
    def load_pickle(filename):
        return joblib.load(filename)


# ### Classifcation with Navie Bayes, SVM and Random Forest

# In[ ]:


class SMSClassification(SMSBase):
    __pipelines = {}
    __params = {}
    __format_model_file_name = '{}_model.pkl'

    def __init__(self, filename, frac=0.8):
        super().__init__(filename, frac)
        
        self.__bow = CountVectorizer(analyzer=self.create_lemmas)
        self.__tfidf = TfidfTransformer()
        
        self.__svd = TruncatedSVD(n_components=50)

        self.__cv = StratifiedKFold(n_splits=10)
        
        self.__default_params = {
            'tfidf__use_idf': (True, False),
            'bow__analyzer': (self.create_lemmas, self.create_tokens),
        }
        
        self.__X = self._df_train['context'].values.tolist()
        self.__y = self._df_train['class'].values.tolist()
        
   
    def __create_pipeline(self, option='NB'):
                        
        if (option in self.__pipelines) is False:
                        
            if option is 'NB':
                classifier = MultinomialNB()
                pipeline = Pipeline([
                    ('bow', self.__bow),
                    ('tfidf', self.__tfidf),
                    ('classifier', classifier),
                ])

            elif option is 'SVM':
                classifier = SVC()
                pipeline = Pipeline([
                    ('bow', self.__bow),
                    ('tfidf', self.__tfidf),
                    ('svd', self.__svd),
                    ('classifier', classifier),
                ])
                
            elif option is 'RFT':
                classifier = RandomForestClassifier()
                pipeline = Pipeline([
                    ('bow', self.__bow),
                    ('tfidf', self.__tfidf),
                    ('svd', self.__svd),
                    ('classifier', classifier),
                ])
                
            else:
                classifier = MultinomialNB()

            self.__pipelines[option] = pipeline
            
            return pipeline

        else:
            return self.__pipelines[option]
            
            
    def __create_grid_search_params(self, option='NB'):
        
        if (option in self.__params) is False:
            if option is 'SVM':
                params = [
                    {
                      'classifier__C': [1, 10, 100, 1000], 
                      'classifier__kernel': ['linear']
                    },
                    {
                      'classifier__C': [1, 10, 100, 1000], 
                      'classifier__gamma': [0.001, 0.0001], 
                      'classifier__kernel': ['rbf']
                    },
                ]

                # merging two list of paramaters on the same list.
#                 params = list(map(lambda m : {**m, **self.__default_params}, params))
            else:
                params = self.__default_params

            self.__params[option] = params
        else:
            params = self.__params[option]
            
        return params

        
        
    def validate(self, option='NB'):
        
        pipeline = self.__create_pipeline(option)
        if pipeline is not None:            
            scores = cross_val_score(pipeline, 
                                     self.__X, 
                                     self.__y, 
                                     scoring='accuracy', 
                                     cv=self.__cv, 
                                     verbose=1, 
                                     n_jobs=-1)

            print('scores={}\nmean={} std={}'.format(scores, scores.mean(), scores.std()))
        else:
            print ("pipeline does not exist!")

        
    def train(self, option='NB', dump=True):
        
        pipeline = self.__create_pipeline(option)
        if pipeline is not None:
            
            params = self.__create_grid_search_params(option)
            
            grid = GridSearchCV(
                pipeline, 
                params, 
                refit=True, 
                n_jobs=-1, 
                scoring='accuracy', 
                cv=self.__cv)

            model = grid.fit(self.__X, self.__y)
            
            display('(Grid Search) Best Parameters:', )
            display(pd.DataFrame([model.best_params_]))

            if dump:
                model_file_name = self.__format_model_file_name.format(option)
                Util.dump_pickle(model, model_file_name)
                
            return model
                
        else:
            print('pipeline does not exist!')
            return None

    
    def test(self, X=None, model=None, model_file=None):
        
        if X is None:
            X = self.__X
        
        if model is None and model_file is None:
            print('Please, use either model or model_file')
            return []
        
        if model_file is not None and os.path.isfile(model_file):
            model = Util.load_pickle(model_file)
            print('{} file was loaded'.format(model_file))
            return model.predict(X)
        
        if model is not None:
            return model.predict(X)
        else:
            return []


# In[12]:


sms = SMSClassification('SMSSpam')


# In[13]:


sms.describe()


# In[27]:


Util.plot_class_dist(sms._df_raw, 'class')


# In[28]:


n_words_in_context = sms._df_raw['n_words'].values.tolist()

figure, axes = plt.subplots(1, 2, figsize=(15,5))
Util.plot_cdf(n_words_in_context, 
         axes[0], 
         xlim=[0, np.mean(n_words_in_context) + 3 * np.std(n_words_in_context) + 50],
         deltay = 0.05,
         ylim=[0, 1.00], xlabel='number of words')

Util.plot_boxplot(sms._df_raw, 'class', 'n_words', axes[1])


# In[29]:


len_of_context = sms._df_raw['len'].values.tolist()

figure, axes = plt.subplots(1, 2, figsize=(15,5))

Util.plot_cdf(len_of_context, 
         axes[0], 
         xlim=[0, np.mean(len_of_context) + 3 * np.std(len_of_context) + 50],
         deltay = 0.05,
         ylim=[0, 1.00], xlabel='len of context')

Util.plot_boxplot(sms._df_raw, 'class', 'len', axes[1])


# + Filtering context by minimum number of words == 1

# In[30]:


sms._df_raw[sms._df_raw['n_words'] == 1]['context'].values.tolist()


# + Filtering context by mean number of words == 16

# In[31]:


sms._df_raw[sms._df_raw['n_words'] == 16]['context'].values.tolist()


# + Filtering context by length of words == 18

# In[33]:


sms._df_raw[sms._df_raw['len'] == 18]['context'].values.tolist()


# + Filtering context by median length of words == 60

# In[34]:


sms._df_raw[sms._df_raw['len'] == 60]['context'].values.tolist()


# ### Validation

# In[4]:


classifiers = ['NB', 'SVM', 'RFT']


# In[33]:


for c in classifiers:
    display('-' * 40 + c + '-' * 40)
    get_ipython().run_line_magic('time', 'sms.validate(c)')


# ### Training

# In[36]:


models = {}

for c in classifiers:
    display('-' * 40 + c + '-' * 40)
    get_ipython().run_line_magic('time', 'models[c] = sms.train(c)')


# ### Testing Stage

# In[37]:


for c in classifiers:
    display('-' * 40 + c + '-' * 40)
    model_file = '{}_model.pkl'.format(c)
    get_ipython().run_line_magic('time', 'r = sms.test(model_file=model_file)')
    display(r)


# ### Performance Evaluation

# In[40]:


for c in classifiers:
    display('-' * 40 + c + '-' * 40)
    model_file = '{}_model.pkl'.format(c)
    model = Util.load_pickle(model_file)
    Util.report_classification(model, 
                               sms._df_train, 
                               sms._df_test, 
                               'context', 
                               'class')


# ### Classification with Deep Learning Algorithm

# In[14]:


import gensim, logging

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.models import Sequential

from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras import models

class SMSDL(SMSBase):
    
    # To implement a processing chain, some global variables were used.
    __clean_tokens = None
    __clean_corpus = None
    __word2vec = None
    __tokenizer = None
    __x_train_token_padded_seqs = None  # pad_sequences
    __x_test_token_padded_seqs = None  # pad_sequences
    __embedding_matrix = None
    __nn_layers = None
    
    # To create word2vec, the parameters are used in Gensim, below.
    # There are a few different approaching like the below instead of creating our word2vec model,
    # + Google's pre-trained vectors based on GoogleNews
    # + GLoVe's pre-trained vectors based on Wikipages
    # + Spacy pre-trained vectors
    __embedding_dim = 300
    __window = 15
    __workers = 4
    __cbow_mean = 1
    __alpha = 0.05
    
    # Creating file names for models with respect to given parameters, above.
    __format_word2vec_model = 'emb_dim:{}_window:{}_cbow:{}_apha:{}.bin'
    __word2vec_file = __format_word2vec_model.format(__embedding_dim, __window, __cbow_mean, __alpha)

    # Creating an embedding matrix using by word2vec model and the parameters, below
    __embedding_vector_length = 300
    __max_nb_words = 200000
    __max_input_length = 50

    # Deep Learning Layers' parameters are using to build a deep network. Our network consists of the layers, below:
    # + Embedding Layer
    # + Dense Layer
    # + LSTM for RNN Layer
    # + Dense Layer
    __num_lstm = 100
    __num_dense = 300
    __rate_drop_out = 0.1
    __rate_drop_lstm = float(0.15 + np.random.rand() * 0.25)
    __rate_drop_dense = float(0.15 + np.random.rand() * 0.25)

    # Creating file names for models with respect to given parameters, above.
    __format_dl_model = 'lstm_%d_%d_%.2f_%.2f.h5'
    __model_dl_file = __format_dl_model % (__num_lstm, __num_dense, __rate_drop_lstm, __rate_drop_dense)

    # In training step, those parameters are using, below.
    __number_of_epochs = 100
    __batch_size = 2048
    __validation_split = 0.1
    __shuffle = True
    
    def __init__(self, filename, frac=0.8):
        super().__init__(filename, frac)
        
        self.__x_name = 'context'
        self.__y_name = 'class'
        
        self.__label_classes = {'ham':0, 'spam':1}
        self.__num_classes = len(self.__label_classes)
        self.__encode_labels()
        
        self.__split_sentence_by_lemmas()
        self.__split_sentence_by_tokens()
            

    def __split_sentence_by_lemmas(self):
        self.__sentences_by_lemmas = list(map(lambda c : self.create_lemmas(c), self._df_raw[self.__x_name].values.tolist()))

    def __split_sentence_by_tokens(self):
        self.__sentences_by_tokens = list(map(lambda c : self.create_tokens(c), self._df_raw[self.__x_name].values.tolist()))

    def __encode_labels(self):
        # https://keras.io/utils/#to_categorical
        encoded_list = list(map(lambda c : self.__label_classes[c], self._df_train[self.__y_name].values.tolist()))
        self.__y_train_one_hot = to_categorical(encoded_list, self.__num_classes)
        
    def __create_word2vec(self, by='lemmas'):
        
        if not os.path.exists(self.__word2vec_file) or self.__word2vec is None:
            if by is 'lemmas':
                sentences = self.__sentences_by_lemmas
            elif by is 'tokens':
                sentences = self.__sentences_by_tokens
            else:
                print('You picked wrong function. Please, check your parameter you are using')
                return
            
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
            
            print('{} doesn\'t exist. A new word2vec is being built...'.format(self.__word2vec_file))
            
            self.__word2vec = gensim.models.Word2Vec(sentences,
                                                    size=self.__embedding_dim,
                                                    window=self.__window,
                                                    workers=self.__workers,
                                                    cbow_mean=self.__cbow_mean,
                                                    alpha=self.__alpha)
            self.__word2vec.save(self.__word2vec_file)

        elif self.__word2vec is not None:
            print('{} has already loaded for word2vec...'.format(self.__word2vec_file))
        else:
            print('{} is loading for word2vec...'.format(self.__word2vec_file))
            self.__word2vec = gensim.models.Word2Vec.load(self.__word2vec_file)
        
    def __create_tokenizer(self):
        
        self.__x_train_corpus = self._df_train[self.__x_name].values.tolist() 
        self.__x_test_corpus = self._df_test[self.__x_name].values.tolist()
        
        all_corpus = self.__x_train_corpus + self.__x_test_corpus

        print('x_train_corpus: {}'.format(len(self.__x_train_corpus)))
        print('x_test_corpus: {}'.format(len(self.__x_test_corpus)))

        # https://keras.io/preprocessing/text/#tokenizer
        self.__tokenizer = Tokenizer(num_words=self.__max_nb_words)
        self.__tokenizer.fit_on_texts(all_corpus)

        print('Found %s unique tokens' % len(self.__tokenizer.word_index))


    def __create_sequences(self):

        # https://keras.io/preprocessing/text/#text_to_word_sequence
        x_train_token_seqs = self.__tokenizer.texts_to_sequences(self.__x_train_corpus)
        x_test_token_seqs = self.__tokenizer.texts_to_sequences(self.__x_test_corpus)

        print('x_train_token_seqs: {}'.format(len(x_train_token_seqs)))
        print('x_test_token_seqs: {}'.format(len(x_test_token_seqs)))
        
        # https://keras.io/preprocessing/sequence/#pad_sequences
        self.__x_train_token_padded_seqs = pad_sequences(x_train_token_seqs, maxlen=self.__max_input_length)
        self.__x_test_token_padded_seqs = pad_sequences(x_test_token_seqs, maxlen=self.__max_input_length)
        print('x_train_token_padded_seqs: {}'.format(self.__x_train_token_padded_seqs.shape))
        print('x_test_token_padded_seqs: {}'.format(self.__x_test_token_padded_seqs.shape))

    def __create_embedding_matrix(self):
        
        token_index = self.__tokenizer.word_index
        self.__number_words = min(self.__max_nb_words, len(token_index)) + 1
        
        self.__embedding_matrix = np.zeros((self.__number_words, self.__embedding_vector_length))
        for word, i in token_index.items():
            if word in self.__word2vec.wv.vocab:
                self.__embedding_matrix[i] = self.__word2vec.wv.word_vec(word)

        print('Null word embeddings: %d' % np.sum(np.sum(self.__embedding_matrix, axis=1) == 0))
        print('embedding_matrix: {}'.format(self.__embedding_matrix.shape))
        
    def __init_weights(self, shape, dtype=None):
        print('init_weights shape: {}'.format(shape))
        # assert  shape == embedding_matrix.shape
        return self.__embedding_matrix

    def __create_embedding_layer(self):
        
        # https://keras.io/layers/embeddings/
        embedding = Embedding(self.__number_words,
                                self.__embedding_vector_length,
                                input_length=self.__max_input_length,
                                mask_zero=True,
                                embeddings_initializer=self.__init_weights)
        
        return embedding

    
    def __create_nn_layers(self, weights_filename=None):

        if self.__nn_layers is None:

            self.__nn_layers = Sequential()
            self.__nn_layers.add(self.__create_embedding_layer())

            # https://keras.io/layers/core/#dense
            # https://keras.io/layers/core/#activation
            self.__nn_layers.add(Dense(self.__num_dense, activation='sigmoid'))
            self.__nn_layers.add(Dropout(self.__rate_drop_out))

            # https://keras.io/layers/recurrent/
            self.__nn_layers.add(LSTM(self.__num_lstm, 
                               dropout=self.__rate_drop_lstm, 
                               recurrent_dropout=self.__rate_drop_lstm))
            
            self.__nn_layers.add(Dense(self.__num_classes, activation='softmax'))

            # https://keras.io/metrics/
            self.__nn_layers.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
            
            self.__nn_layers.summary()
        
        if weights_filename is not None:
            self.__nn_layers.load_weights(weights_filename)

    def __build_model(self, weights_filename=None):

        self.__create_word2vec()
        self.__create_tokenizer()
        self.__create_sequences()
        self.__create_embedding_matrix()
        self.__create_nn_layers(weights_filename)

    def __create_callbacks(self, tensorboard):

        callbacks = []
        # https://keras.io/callbacks/#usage-of-callbacks
        early_stopping = EarlyStopping(monitor='val_acc', patience=10)

        print(self.__model_dl_file)
        # https://keras.io/callbacks/#modelcheckpoint
        model_checkpoint = ModelCheckpoint(self.__model_dl_file, save_best_only=True)

        # https://keras.io/callbacks/#tensorboard
        if tensorboard:
            tensor_board = TensorBoard(log_dir='./logs',
                                       histogram_freq=5,
                                       write_graph=True,
                                       write_images=True,
                                       embeddings_freq=0,
                                       embeddings_layer_names=None,
                                       embeddings_metadata=None)
            callbacks.append(tensor_board)

        callbacks.append(early_stopping)
        callbacks.append(model_checkpoint)

        return callbacks
    
    def load_model(self, filename):
        model = models.load_model(filename)
        return model

    def train(self, tensorboard_enable=False):

        self.__build_model()

        callbacks = self.__create_callbacks(tensorboard_enable)

        # https://keras.io/models/model/
        self._model = self.__nn_layers.fit(self.__x_train_token_padded_seqs,
                             self.__y_train_one_hot,
                             epochs=self.__number_of_epochs,
                             batch_size=self.__batch_size,
                             validation_split=self.__validation_split,
                             shuffle=self.__shuffle,
                             callbacks=callbacks)

        best_val_score = max(self._model.history['val_acc'])
        print('Best Score by val_acc: {}'.format(best_val_score))
        
        self.__df_history = pd.DataFrame()
        self.__df_history['acc'] = self._model.history['acc']
        self.__df_history['loss'] = self._model.history['loss']
        self.__df_history['val_acc'] = self._model.history['val_acc']
        self.__df_history['val_loss'] = self._model.history['val_loss']
        
    
    def __test(self, X_token_padded_seqs):
        prediction_probs = self.__nn_layers.predict(X_token_padded_seqs,
                                             batch_size=self.__batch_size,
                                             verbose=1)

        pre_label_ids = list(map(lambda probs: probs.argmax(), list(prediction_probs)))
        classes = ['ham', 'spam']
    
        return list(map(lambda x: classes[x], pre_label_ids))        
    
            
    def test(self, X_token_padded_seqs=None, weights_filename=None):
        self.__build_model(weights_filename=weights_filename)
        
        if X_token_padded_seqs is None:
            return self.__test(self.__x_test_token_padded_seqs)
        else:
            return self.__test(X_token_padded_seqs)
        
    def __test2(self, X_token_padded_seqs, model):
        prediction_probs = model.predict(X_token_padded_seqs,
                                             batch_size=self.__batch_size,
                                             verbose=1)

        pre_label_ids = list(map(lambda probs: probs.argmax(), list(prediction_probs)))
        classes = ['ham', 'spam']

        return list(map(lambda x: classes[x], pre_label_ids))  
    
    def report_cm2(self, model):        
        X_train = self.__x_train_token_padded_seqs
        X_test = self.__x_test_token_padded_seqs
        
        y_train = self._df_train[self.__y_name].values.tolist()
        y_test = self._df_test[self.__y_name].values.tolist()
        
        y_train_pred = self.__test2(self.__x_train_token_padded_seqs, model)
        y_test_pred = self.__test2(self.__x_test_token_padded_seqs, model)
        
        classes = ['ham', 'spam']
        
        Util.report_cm(y_train, y_test, y_train_pred, y_test_pred, classes)
    
    def report_cm(self, weights_filename):
        self.__build_model(weights_filename)
        
        X_train = self.__x_train_token_padded_seqs
        X_test = self.__x_test_token_padded_seqs
        
        y_train = self._df_train[self.__y_name].values.tolist()
        y_test = self._df_test[self.__y_name].values.tolist()
        
        y_train_pred = self.__test(self.__x_train_token_padded_seqs)
        y_test_pred = self.__test(self.__x_test_token_padded_seqs)
        
        classes = ['ham', 'spam']
        
        Util.report_cm(y_train, y_test, y_train_pred, y_test_pred, classes)

    def display_history(self):
        display(self.__df_history.describe())
        
    def plot_acc(self):
        self.__df_history[['acc', 'val_acc']].plot()
        
    def plot_loss(self):
        self.__df_history[['loss', 'val_loss']].plot()


# In[15]:


sms_dl = SMSDL('SMSSpam')


# In[23]:


tokenizer = sms_dl._SMSDL__tokenizer
freq_of_unique_words = list(map(lambda x : x[1], tokenizer.word_counts.items()))

figure, axes = plt.subplots(1, figsize=(15,5))
Util.plot_cdf(freq_of_unique_words, 
              axes,
              xlim = [0, 50],
              deltay = 0.05,
              ylim = [0, 1.00], xlabel='freq of unique words')


# In[106]:


sim = {}
sample_words = ['phone', 'sms', 'bank', 
                'call', 'discount', 'off', 
                'award', 'winner', 'free', 
                'text', 'cash', 'money', 
                'credit', 'prize', 'insurance', 
                'sale', 'click', 'subscriber', 
                '%', '$', 'pound']

for w in sample_words:
    sim[w] = sms_dl._SMSDL__word2vec.most_similar(w)

pd.DataFrame(sim).applymap(lambda x: '{} : {:.2f}'.format(x[0], float(x[1]))).T


# ### Training

# In[7]:


sms_dl.train()


# In[40]:


sms_dl.display_history()


# ### Plotting Accuracy and Learning Curves

# In[20]:


sms_dl.plot_acc()
sms_dl.plot_loss()


# ### Performance Evaluation

# In[21]:


sms_dl.report_cm('lstm_100_300_0.33_0.30.h5')


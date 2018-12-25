# -*- coding: utf-8 -*-

import json
import pandas as pd
import ast
import matplotlib.pyplot as plt 
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.optimizers import Adam, Adamax, Nadam, RMSprop
from keras.callbacks import EarlyStopping

path = 'E:\\ML\\Datasets\\yelp_dataset\\'
file = 'yelp_academic_dataset_review.json'

def read_convert_json(path, file, to_file):
    """Read and transforms json file to pandas dataframe and csv file"""
    with open (f'{path}{file}', 'rb') as f:
        data = f.readlines()
    data = [i.rstrip() for i in data] #Removing \n at the end of the line
    data = [i.decode('UTF-8') for i in data] #decoding data with 'UTF-8'
    data_dicts = [ast.literal_eval(i) for i in data] # transform data to list of dicts
    dataset = pd.DataFrame(data_dicts)
    dataset.to_csv(f'{path}{to_file}', sep=';', index=False)
    return dataset

def building_subset(X_train, Xtrain, X_processed, word2vec, ytrain, y_train):
    for train_ind, index_ in enumerate(X_train):
        tokens = X_processed[index_]
        for token_id, token in enumerate(tokens):
            if token not in word2vec.wv:
                pass
            else:
                Xtrain[train_ind, token_id, :] = word2vec.wv[str(token)]
        if y_train[index_] == 0.0:
            ytrain[train_ind, :] = [1., 0., 0.]
        elif y_train[index_] == 1.0:
            ytrain[train_ind, :] = [0., 1., 0.]
        else:
            ytrain[train_ind, :] = [0., 0., 1.]
    return Xtrain, ytrain

def building_subset_after_load(X_train, Xtrain, X_processed, word2vec, ytrain, y_train):
    for train_ind, index_ in enumerate(X_train):
        tokens = X_processed[index_]
        for token_id, token in enumerate(tokens):
            if token not in word2vec.wv:
                pass
            else:
                Xtrain[train_ind, token_id, :] = word2vec.wv[str(token)]
        if y_train.loc[train_ind, :].values == 0.0:
            ytrain[train_ind, :] = [1., 0., 0.]
        elif y_train.loc[train_ind, :].values == 1.0:
            ytrain[train_ind, :] = [0., 1., 0.]
        else:
            ytrain[train_ind, :] = [0., 0., 1.]
    return Xtrain, ytrain

def visualize_history(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valditaion'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    
#splitting words to tokens, lemmatization, removing stop words
nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))
nlp = spacy.load('en_core_web_sm')

def tokenize(text):
    tkns = []
    text = re.sub('[^a-zA-Z]', ' ', text)
    doc = nlp(text)
    for token in doc:
        if token.is_digit:
            pass
        elif token.is_space:
            pass
        elif token.is_punct:
            pass
        elif token.is_stop:
            pass
        elif len(token) <= 3:
            pass
        else:
            tkns.append(token.lemma_.lower())
    return tkns

def build_cnn(batch_size, nb_epochs, max_length, vector_size):
    
    model = Sequential()
    model.add(Conv1D(32, kernel_size=3, activation='elu', padding='same', input_shape=(max_length, vector_size)))
    model.add(Dropout(0.25))
    model.add(Conv1D(32, kernel_size=2, activation='elu', padding='same'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='tanh'))
    model.add(Dropout(0.5))
    
    model.add(Dense(3, activation='softmax'))
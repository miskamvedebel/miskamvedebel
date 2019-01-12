# -*- coding: utf-8 -*-

import nltk
nltk.download('stopwords')
import spacy
import re
from gensim.models.word2vec import Word2Vec
from keras.models import load_model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.optimizers import Adam, Adamax, Nadam, RMSprop
from keras.callbacks import EarlyStopping
import numpy as np

#splitting words to tokens, lemmatization, removing stop words
def tokenize(text):
    tkns = []
    nlp = spacy.load('en_core_web_sm')
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
        else:
            tkns.append(token.lemma_.lower())
    return tkns

def vectorize(tokens, word2vec, vector):
    for tkn_id, token in enumerate(tokens[:81]):
        if token not in word2vec.wv:
            continue
        else:
            vector[0, tkn_id, :] = word2vec.wv[str(token)]
    return vector

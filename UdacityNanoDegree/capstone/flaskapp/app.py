# -*- coding: utf-8 -*-
from flask import Flask, render_template, Response, request, redirect, url_for
from models.helpers import *
from keras.backend import clear_session
import tensorflow as tf

word2vec = Word2Vec.load('./models/word2vec_model_120.model')
max_lenght = 81
vector_size = 120
vector = np.zeros(shape=(1, max_lenght, vector_size))

app = Flask(__name__)

@app.route('/')
def index():
    sentiment = 'Here will be sentiment'
    sentence = 'Your sentence'
    tokenized = 'Tokenized representation'
    return render_template('index.html', sentence=sentence, tokenized=tokenized, sentiment=sentiment)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/analyze/', methods=['GET','POST'])
def test():

    text = request.form['text']
    tokens = tokenize(text)
    vector_ = vectorize(tokens, word2vec, vector)
    clear_session()
    prediction = load_model('./models/CNN_120.h5').predict_classes(vector_)
    if prediction == [0]:
        sentiment = 'Negative'
    elif prediction == [1]:
        sentiment = 'Neutral'
    else:
        sentiment = 'Positive'
    return render_template('index.html', sentiment=sentiment, sentence=text, tokenized=tokens)

if __name__ == '__main__':
    app.run(debug=True)

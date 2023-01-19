import keras
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
import pandas as pd
import tensorflow as tf
import re
from keras.callbacks import ModelCheckpoint


def predict(review):
    data = pd.read_csv("train_dataset.csv", nrows=1000)
    data['Sentiment'] = data['Sentiment'].map({1: 0, 2: 1})
    inputs = data['Text'].str.replace('[^a-zA-Z ]', '')
    labels = data['Sentiment']

    test_data = pd.read_csv("test_dataset.csv", nrows=200)
    test_data['Sentiment'] = test_data['Sentiment'].map({1: 0, 2: 1})
    test_inputs = test_data['Text'].str.replace('[^a-zA-Z ]', '')
    test_labels = test_data['Sentiment']

    training_sentences = []
    training_labels = []
    testing_sentences = []
    testing_labels = []
    for row in inputs:
        training_sentences.append(str(row))
    for row in labels:
        training_labels.append(row)
    for row in test_inputs:
        testing_sentences.append(str(row))
    for row in test_labels:
        testing_labels.append(row)

    tokenizer = Tokenizer(num_words=40000, oov_token="<OOV>")
    tokenizer.fit_on_texts(inputs)

    model = Sequential()
    model.add(Embedding(input_dim=40000, output_dim=16, input_length=128))
    model.add(LSTM(units=16, dropout=0.5))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    sequences = tokenizer.texts_to_sequences(training_sentences)
    input_data = pad_sequences(sequences, maxlen=128, truncating='post')

    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    test_data = pad_sequences(testing_sequences, maxlen=128, truncating='post')

    labels = np.array(labels)
    test_labels = np.array(test_labels)

    checkpoint1 = ModelCheckpoint("best_sent_model.hdf5", monitor='val_accuracy', verbose=1, save_best_only=True,
                                  mode='max')

    model.fit(input_data, labels, epochs=20, validation_data=(test_data, test_labels),
              callbacks=[checkpoint1], shuffle=True, validation_split=0.2)

    new_review = [review]
    review_seq = tokenizer.texts_to_sequences(new_review)
    review_pad = pad_sequences(review_seq, maxlen=128)

    prediction = model.predict(review_pad)

    return prediction[0]

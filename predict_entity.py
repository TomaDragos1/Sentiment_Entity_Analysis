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
    inputs = data['Text'].str.replace('[^a-zA-Z ]', '')
    entity_mapping = {'movie': 0, 'place': 1, 'app': 2, 'product': 3}
    data['Entity'] = data['Entity'].map(entity_mapping)
    labels = data['Entity']

    test_data = pd.read_csv("test_dataset.csv", nrows=200)
    test_inputs = test_data['Text'].str.replace('[^a-zA-Z ]', '')
    entity_mapping = {'movie': 0, 'place': 1, 'app': 2, 'product': 3}
    test_data['Entity'] = test_data['Entity'].map(entity_mapping)
    test_labels = test_data['Entity']

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
    model.add(Dense(units=4, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    sequences = tokenizer.texts_to_sequences(training_sentences)
    input_data = pad_sequences(sequences, maxlen=128, truncating='post')

    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    test_data = pad_sequences(testing_sequences, maxlen=128, truncating='post')

    labels = tf.keras.utils.to_categorical(labels, num_classes=4)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=4)

    labels = np.array(labels)
    test_labels = np.array(test_labels)

    checkpoint1 = ModelCheckpoint("best_ent_model.hdf5", monitor='val_accuracy', verbose=1, save_best_only=True,
                                  mode='max')

    model.fit(input_data, labels, epochs=20, validation_data=(test_data, test_labels),
              callbacks=[checkpoint1], validation_split=0.2, shuffle=True)

    reviews = [review]
    review_seq = tokenizer.texts_to_sequences(reviews)
    review_pad = pad_sequences(review_seq, maxlen=128)

    prediction = model.predict(review_pad)

    return prediction[0]

# there are three main steps:
# 1. preprocessing
# 2. word2vec embedding layer
# 3. training keras sequential model
import pandas as pd
from numpy import array
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split

import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
from nltk import word_tokenize

def pre_processing(filename='train_data.csv'):
    train_data = pd.read_csv(filename, index_col=0)
    print(train_data.head())
    print(train_data.describe())
    print(train_data.info())
    # set seaborn style
    # sns.set(style="whitegrid")
    nltk.download('punkt')
    nltk.download('stopwords')
    en_stopwords = stopwords.words('english')
    # Detokenizer combines tokenized elements
    detokenizer = TreebankWordDetokenizer()

    def clean_description(desc):
        desc = word_tokenize(desc.lower())
        desc = [token for token in desc if token not in en_stopwords and token.isalpha()]
        return detokenizer.detokenize(desc)

    train_data['review'] = train_data['features'].apply(clean_description)
    train_data.dropna(inplace=True)

#     # target_1_values = set(df['continuous_target_1'])
#     # >> > target_1_values
#     # {80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100}
    def points_to_class(points):
        if points in range(80, 83):
            return 0
        elif points in range(83, 87):
            return 1
        elif points in range(87, 90):
            return 2
        elif points in range(90, 94):
            return 3
        elif points in range(94 - 98):
            return 4
        else:
            return 5
    print(train_data.columns)
    train_data['label'] = train_data['continuous_target_1'].apply(points_to_class)
    return train_data[["review","label"]]

def train_model(data):
    X = data['review']
    Y = data['label']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    # getting total number of words
    num_words = len([word for sent in X for word in sent.split(' ')])
    # getting the biggest sentence length for padding
    max_length = max([len(sent.split()) for sent in X_train])
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(X_train)
    sequences_train = tokenizer.texts_to_sequences(X_train)
    x_train_seq = pad_sequences(sequences_train, maxlen=max_length)
    sequences_test = tokenizer.texts_to_sequences(X_test)
    x_test_seq = pad_sequences(sequences_test, maxlen=max_length)

    #x_val_seq = x_train_seq[-30000:]
    #x_train_seq = x_train_seq[:-30000]
    #y_val = Y_train[-30000:]
    #Y_train = Y_train[:-30000]
    # specifying CNN for multi-label classification 
    model = tf.keras.Sequential()
    embedding_layer = tf.keras.layers.Embedding(num_words, 50, input_length=max_length)
    model.add(embedding_layer)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(56, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.summary()
    #'binary_cross_entropy' and 'adam' did not work, dont know why
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #batch_size = 8
    #steps_per_epoch = int(len(x_train_seq) / batch_size)
    #steps_per_validation = int(len(x_val_seq) / batch_size)
    #history = model.fit(x_train_seq, Y_train, steps_per_epoch=steps_per_epoch, epochs=2,validation_data=(x_val_seq, y_val),
    #                    validation_steps=steps_per_validation,  batch_size=batch_size, verbose=2)
    history = model.fit(x_train_seq, Y_train)
    print('Starting to fit the model')
    #history = model.fit(x_train_seq, Y_train, epochs=20, verbose=2,
    #        validation_data = (x_test_seq, Y_test), batch_size = 32)

    # Save the model
    model.save('sentiment_analysis.h5')
    loss, accuracy = model.evaluate(x_train_seq, Y_train, verbose = False)
    print("Training accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(x_test_seq, Y_test, verbose = False)
    print("Testing accuracy: {:.4f}".format(accuracy))

if __name__ == '__main__':
    data = pre_processing()
    train_model(data)

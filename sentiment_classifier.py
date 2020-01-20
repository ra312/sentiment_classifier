# there are three main steps:
# 1. preprocessing
# 2. loading of GloVe
# 2. keras sequential model = embedding+CNN
# 3. training keras sequential model
# 4. evaluation
# 5. goodness of fit test

GPL_lines = '\n'.join([
    "Wine reviews sentiment classification",
    "Copyright (C) 2020 Rauan Akylzhanov",
    'This program comes with ABSOLUTELY NO WARRANTY; for details type show w',
    "This is free software, and you are welcome to redistribute it under certain conditions; type `show c' for "
    "details. "
])
print(GPL_lines)

import os

import nltk
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

# set parameters
batch_size = 32
embedding_dim = 50
filters = 250
kernel_size = 3
hidden_dim = 250
epochs = 2
GLOVE = 'glove.6B'
GLOVE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), GLOVE)

VALIDATION_SPLIT = 0.2
MAX_SEQUENCE_LENGTH = 3000
EMBEDDING_DIM = 100


def pre_processing(filename='data.csv'):
    print('loading and tokenizing data ....')
    train_data = pd.read_csv(filename, index_col=0)
    print(train_data.head())
    print(train_data.describe())
    print(train_data.info())
    nltk.download('punkt')
    nltk.download('stopwords')
    en_stopwords = stopwords.words('english')
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

    train_data['label'] = train_data['continuous_target_1'].apply(points_to_class)

    return train_data[["review", "label"]]


def load_glove():
    # checking if glove.6B folder exists in the current directory

    glove_exception = Exception('glove.6B files are missing or empty')
    try:
        if os.path.exists(GLOVE_DIR) and os.path.isdir(GLOVE_DIR):
            for n in [50, 100, 200]:
                glove_file = GLOVE + '.' + str(n) + 'd.txt'
                glove_file_path = os.path.join(GLOVE_DIR, glove_file)
                if os.path.isfile(glove_file_path):
                    if os.path.getsize(glove_file_path) == 0:
                        raise glove_exception
                else:
                    raise glove_exception
        else:
            raise glove_exception
    except Exception as glove_exception:
        print(glove_exception)
        print('loading GLOVE.6B files')
        os.system('./load_glove.sh')


def get_embeddings_index():
    load_glove()
    print('Indexing word vectors.')
    embeddings_index = {}
    with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embeddings_index[word] = coefs
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index


def train_model(df):
    embeddings_index = get_embeddings_index()
    reviews = df['review']
    labels = df['label']
    num_words = len([word for sentence in reviews for word in sentence.split(' ')])
    # getting the biggest sentence length for padding
    max_num_words = max([len(sentence.split()) for sentence in reviews])
    tokenizer = Tokenizer(num_words=max_num_words)
    tokenizer.fit_on_texts(reviews)
    sequences = tokenizer.texts_to_sequences(reviews)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = to_categorical(np.asarray(labels))

    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    # split the data into a training set and a validation set
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    x_train, x_test, y_train, y_test = train_test_split(data, labels,
                                                        test_size=0.2, random_state=1)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                      test_size=0.2, random_state=1)

    # num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
    #
    # x_train = data[:-num_validation_samples]
    # y_train = labels[:-num_validation_samples]
    # x_val = data[-num_validation_samples:]
    # y_val = labels[-num_validation_samples:]

    print('Preparing embedding matrix.')

    # prepare embedding matrix
    num_words = min(max_num_words, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= max_num_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    print('Training model.')

    # train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    num_classes = len(set(df['label']))
    num_classes = 6
    preds = Dense(num_classes, activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    history = model.fit(x_train, y_train,
                        batch_size=128,
                        epochs=5,
                        validation_data=(x_val, y_val))
    model.save(filepath='sentiment.h5', overwrite=True, save_format=True)
    # Evaluate the model on the test data using `evaluate`
    print('\n# Evaluate on test data')
    results = model.evaluate(x_test, y_test, batch_size=128)
    print('test loss, test acc:', results)

    return history


if __name__ == '__main__':
    df = pre_processing()
    history = train_model(df=df)

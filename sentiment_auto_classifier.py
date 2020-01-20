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
#from tensorflow.keras.initializers import Constant
#from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding
#from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D
#from tensorflow.keras.models import Model
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.optimizers import Adam, SGD
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

    # train_data['label'] = train_data['continuous_target_1'].apply(points_to_class)

    train_data['label'] = train_data['continuous_target_1']
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
    # # getting the biggest sentence length for padding
    max_num_words = max([len(sentence.split()) for sentence in reviews])
    tokenizer = Tokenizer(num_words=max_num_words)
    tokenizer.fit_on_texts(reviews)
    x_train = tokenizer.texts_to_sequences(reviews)
    # x_train = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    # y_train = to_categorical(np.asarray(labels))
    y_train = np.asarray(labels)
    word_index = tokenizer.word_index
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    id_to_word = {value: key for key, value in word_index.items()}
    # Convert the word indices to words.
    validation_split = 0.2
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
                                                        test_size=validation_split, random_state=1)

    x_train = list(map(lambda sentence: ' '.join(
        id_to_word[i] for i in sentence), x_train))
    x_test = list(map(lambda sentence: ' '.join(
        id_to_word[i] for i in sentence), x_test))
    x_train = np.array(x_train, dtype=np.str)
    x_test = np.array(x_test, dtype=np.str)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    # print('Found %s unique tokens.' % len(word_index))
    # data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    # labels = to_categorical(np.asarray(labels))
    #
    # print('Shape of data tensor:', data.shape)
    # print('Shape of label tensor:', labels.shape)
    #
    # # split the data into a training set and a validation set
    # indices = np.arange(data.shape[0])
    # np.random.shuffle(indices)
    # data = data[indices]
    # labels = labels[indices]
    # x_train, x_test, y_train, y_test = train_test_split(data, labels,
    #                                                     test_size=0.2, random_state=1)
    #
    # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
    #                                                   test_size=0.2, random_state=1)

    import autokeras as ak
    text_classifier = ak.TextClassifier(max_trials = 10)
    # x = np.asarray(df['review'].values)
    # x = np.asarray(list(map(np.str_, x)))
    # y = np.asarray(df['label'].values)


    text_classifier.fit(x_train,y_train, epochs = 5)
    predicted_y = clf.predict(x_test)
    # Evaluate the best model with testing data.
    print(text_classifier.evaluate(x_test, y_test))



if __name__ == '__main__':
    df = pre_processing()
    model = train_model(df=df)

from keras.models import Sequential
from keras.layers import Embedding, Dropout, Conv1D, MaxPool1D, LSTM, Dense, Activation
from keras.utils import pad_sequences, to_categorical
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer

# from keras.datasets import imdb
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=1000)
f = open('Video_Games.json')

df = pd.read_json(f, lines=True)
df = df[['overall', 'reviewText']]
df = df.dropna()
df = df.reset_index(drop=True)

df = np.array(df)
df = df[0:10]
# preprocess our data
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np

from sklearn.model_selection import train_test_split

# split our data
(x_train, x_test, y_train, y_test) = train_test_split(df[:, 0], df[:, 1], test_size=0.2, random_state=42)

# Tokenize our stuff
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(x_train)
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)

# padding
x_train = pad_sequences(x_train, maxlen=100)
x_test = pad_sequences(x_test, maxlen=100)

# one-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# model, compile, and train

model = Sequential()
model.add(Embedding(len(y_train), 100))
model.add(Dropout(0.5))
model.add(Conv1D(64, 5, padding='valid', activation='relu', strides=1))
model.add(MaxPool1D(pool_size=4))
model.add(LSTM(55))
model.add(Dense(5))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

history = model.fit(x_train, y_train, batch_size=100, epochs=10, validation_data=(x_test, y_test))

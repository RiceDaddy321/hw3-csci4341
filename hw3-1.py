import json
from zipfile import ZipFile

import numpy as np
from keras.utils import pad_sequences, image_dataset_from_directory, get_file
from datasets import load_dataset
import os
import pandas as pd
# import the zip
# os.makedirs("celeba_gan", exist_ok=True)
#
# with ZipFile("celeba_gan/data.zip", "r") as zipobj:
#     zipobj.extractall("celeba_gan")
#
# dataset = image_dataset_from_directory(
#     "celeba_gan", label_mode=None, image_size=(64, 64), batch_size=32
# )
f = open('Video_Games.json')

# df = json.load(json.dump(f))
# df = np.array(df)
df = pd.read_json(f, lines=True)


n = len(df)
train_df = df[0:int(n * 0.7)]
val_df = df[int(n * 0.7):int(n * 0.9)]
test_df = df[int(n * 0.9):]

#import the dataset
(x_train, y_train), (x_test, y_test) = [], []

# padding
x_train = pad_sequences(x_train, maxlen=100)
x_test = pad_sequences(x_test, maxlen=100)

# model, compile, and train
from keras.models import Sequential
from keras.layers import Embedding, Dropout, Conv1D, MaxPool1D, LSTM, Dense, Activation
model = Sequential()
model.add(Embedding(5000, 100))
model.add(Dropout(0.5))
model.add(Conv1D(64, 5, padding='valid', activation='relu', strides=1))
model.add(MaxPool1D(pool_size=4))
model.add(LSTM(55))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

history = model.fit(x_train, y_train, batch_size=100, epochs=10, validation_data=(x_test, y_test))
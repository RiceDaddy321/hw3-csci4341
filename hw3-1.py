import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Embedding, Dropout, Conv1D, MaxPool1D, LSTM, Dense, Activation
from tensorflow.keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

f = open("Video_Games.json")

df = pd.read_json(f, lines=True)
df = df[['overall', 'reviewText']]
df = df.dropna()
df = df.reset_index(drop=True)

df = np.array(df)

f = open("Video_Games.json")

df = pd.read_json(f, lines=True)
df = df[['overall', 'reviewText']]
df = df.dropna()
df = df.reset_index(drop=True)

df = np.array(df)

(y_train, y_test, x_train, x_test) = train_test_split(df[:, 0], df[:, 1], test_size=0.2, random_state=42)

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
model.add(Dense(6, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

history = model.fit(x_train, y_train, batch_size=100, epochs=10, validation_data=(x_test, y_test))

# save it just in case
# model.save('model.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, marker='.', c='blue', label="Train")
plt.plot(epochs, val_acc, marker='.', c='orange', label="Validation")
plt.title('Training and validation accuracy')
plt.legend()

plt.savefig("acc.png")
plt.figure()

plt.plot(epochs, loss, marker='.', c='blue', label="Train")
plt.plot(epochs, val_loss, marker='.', c='orange', label="Validation")
plt.title('Training and validation loss')
plt.legend()

plt.savefig("loss.png")

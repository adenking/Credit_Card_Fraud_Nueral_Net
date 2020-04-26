import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
import pickle
from tensorflow.keras.callbacks import TensorBoard
import time

pickle_in = open('data\\processed\\X.pickle', 'rb')
X = pickle.load(pickle_in)

pickle_in = open('data\\processed\\y.pickle', 'rb')
y = pickle.load(pickle_in)
print(X)

dense_layers = [1, 2, 3, 4, 5]
layer_sizes = [8, 16, 32, 64, 128, 256]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        NAME = "{}-nodes-{}-dense-{}".format(layer_size,
                                             dense_layer, int(time.time()))
        print(NAME)

        model = Sequential()

        model.add(Flatten())

        for _ in range(dense_layer):
            model.add(Dense(layer_size))
            model.add(Activation('relu'))

        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        tensorboard = TensorBoard(log_dir="logs\\{}".format(NAME))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'],
                      )

        model.fit(X, y,
                  batch_size=512,
                  epochs=10,
                  validation_split=0.3,
                  callbacks=[tensorboard])
        model.save(f'models\\{NAME}.model')

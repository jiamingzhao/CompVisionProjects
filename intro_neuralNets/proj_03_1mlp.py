# Jiaming Zhao
# modified mnist_mlp.py from Keras

from __future__ import print_function

import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from sklearn.metrics import confusion_matrix
from scipy.sparse import spdiags

def conf_matrix(model, x_test, y_true):

    # the #i of y_test is the #i of the flat array, if [i][j] == 1 then #j is the #i_val of the flat array (prediction)
    # the real values are stored in y_train_true

    x_pred = model.predict(x_test)

    # take the largest value as the one
    x_pred_flat = []
    for i in range(len(x_pred)):
        x_pred_flat.append( np.argmax(x_pred[i]) )

    cm = confusion_matrix(y_true, x_pred_flat).astype(np.float32)

    x_hlen = len(cm)
    x_wlen = len(cm[0])

    # normalize all values in the confusion matrix in the row as sum of 1
    for i in range(x_hlen):
        summed = 1. * sum( cm[i] )
        for j in range(x_wlen):
            cm[i][j] = cm[i][j] / summed

    return cm


def print_cm(cm):  # prints the cm in decimal notation instead of scientific float notation
    for i in range(len(cm)):
        cm_row = '['
        for j in range(len(cm[0])):
            cm_row += '  ' + '{0:f}'.format(cm[i][j])
        print(cm_row, ' ]')


batch_size = 128
num_classes = 10
epochs = 20

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_test = x_test.reshape(10000, 784)
x_test = x_test.astype('float32')
x_test /= 255

y_test_cpy = np.copy(y_test)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Using Saved Keras model from the following code

x_train = x_train.reshape(60000, 784)
x_train = x_train.astype('float32')
x_train /= 255

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)

# Try using saved Keras model
try:
    model = load_model('mnist_mlp_model.h5')
    print('Using mnist_mlp_model.h5 model from directory')
except:
    print('mnist_mlp_model.h5 has not been found in directory. Training new model...')
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=(784,)))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.summary()

    model.compile(loss='mean_squared_error',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))

    model.save('mnist_mlp_model.h5')
    print('model has been saved as mnist_mlp_model.h5')


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

cm = conf_matrix(model, x_test, y_test_cpy)

print_cm(cm)



"""
MNIST CNN with TReLU
B. Despres
modification of a convent by: [fchollet](https://twitter.com/fchollet)
31/10/2021
Achieves ~99% test accuracy on MNIST.
"""

"""
## Setup
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' 

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K
import matplotlib.pyplot as plt

from keras.optimizers import Adam,SGD
from tensorflow.keras import initializers

import sys



"""
## Prepare the data
"""
# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_test[i])
plt.show()




"""
## Build the model
"""
def T_relu(x):
    return K.relu(x, max_value=1)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation=T_relu),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation=T_relu),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)


model.summary()
#sys.exit(0)


"""
## Train the model
"""
batch_size = 128*8
epochs = 5
#'mse'
#model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.compile(loss="mse", optimizer="SGD", metrics=["accuracy"])
#model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.SGD(momentum=0.1, 
#                                         nesterov=True),
#                                         metrics=["accuracy"])
#model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

"""
## Evaluate the trained model
"""
score = model.evaluate(x_test, y_test, verbose=0)

y_sortie=model.predict(x_test)

print(y_sortie[0])
print(y_test[0])
print("Test loss:", score[0])
print("Test accuracy:", score[1])

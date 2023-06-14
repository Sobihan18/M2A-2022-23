

"""
local system parameters and Setup
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' 
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense


"""
Build a dense NN model
"""
print(" ")
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("Dense NN model")
input_shape=1
model = Sequential()
model.add(Dense(3, input_dim=input_shape,name="hidden1",
                use_bias=True,  
                activation='relu'))
model.add(Dense(3,name="hidden2",
                activation='relu'))
model.add(Dense(3,name="hidden3",
                use_bias=False, 
                activation='relu'))
model.add(Dense(4,name="hidden4",
                use_bias=True,  
                activation='relu'))
model.add(Dense(1,name="out",
                use_bias=True, 
                activation='linear'))

model.summary()

print(" ")
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

"""
## Build a CNN model
"""
print(" ")
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("CNN model with 1 channel/color")

num_classes = 10
input_shape = (28, 28,1)

model2 = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), use_bias=False,padding="same",activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3),use_bias=True,padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model2.summary()


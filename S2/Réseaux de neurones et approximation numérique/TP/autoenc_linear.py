#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 16:55:06 2021

@author: pironneau
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 16:40:40 2021

@author: pironneau
"""
#https://github.com/shibuiwilliam/Keras_Autoencoder
#https://github.com/shibuiwilliam/Keras_Autoencoder/blob/master/Cifar_Conv_AutoEncoder.ipynb
#from tensorflow.keras.models import load_model

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

import os
import numpy as np
#import cv2
import glob

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import numpy as np
import glob

import sys


    
saveDir = " /Users/Despres/Desktop/"

batch_size = 320
num_classes = 10
epochs = 5

print("\n------------------------------------------\n")



# Loading datasets
DATA_DIR = "./DATA/"


#nombre_labels=np.loadtxt(DATA_DIR + "FORMAT_tak.txt", dtype=np.int32)

strstr_train='./DATA/TRAIN_auto.txt'
fichier_format = open(strstr_train,'r')
#infos = fichier_format.readline(-1).split(",")
#print("infos=",infos)
#nombre_labels=int(infos[0])
#nombre_points=int(infos[1])

nombre_label=0
nombre_points=10

print("nombre_points=",nombre_points)





train_data = np.loadtxt(strstr_train, dtype=np.float32, delimiter=',')
#np.random.shuffle(train_data)

fichier_format.close()

print("cou")


x_train = train_data[:, 0:nombre_points]
y_train = x_train

#sys.exit(0)


print ("Taille x_train=",x_train.shape)
print ("Taille y_train=",y_train.shape)


#sys.exit(0)





model = Sequential()
model.add(Dense(8, input_dim=nombre_points,name="W_0",
                kernel_initializer='random_uniform',
                use_bias=False, bias_initializer='random_uniform', 
                activation='linear'))

model.add(Dense(nombre_points,name="W_1",
                kernel_initializer='random_uniform',
                use_bias=False, bias_initializer='random_uniform', 
                activation='linear'))




model.compile(optimizer='adam', loss='mse')

model.summary()

model.fit(x_train, y_train,
		  batch_size=100,
		  epochs=500,
		  verbose=1 #, validation_data=(x_test, y_test)
		  ) 

#sys.exit(0)


# load pretrained weights
#model.load_weights("/Users/pironneau/Desktop/cifar10/AutoEncoder_Cifar10_Deep_weights.05-0.56-0.56.hdf5")

es_cb = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
chkpt = DATA_DIR + 'AutoEncoder_Cifar10_Deep_weights.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5'
cp_cb = ModelCheckpoint(filepath = chkpt, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

history = model.fit(x_train, x_train,
                    batch_size=batch_size,
                    epochs=3,
                    verbose=1,
                    validation_data=(x_train, x_train),
                    callbacks=[es_cb, cp_cb],
                    shuffle=True)

score = model.evaluate(x_train, y_train, verbose=1)
print(score)



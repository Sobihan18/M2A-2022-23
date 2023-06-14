#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 07:35:21 2021

@author: despres
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'



import numpy as np
import tensorflow
import tensorflow.keras as keras
from keras import backend as K

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
# from keras import metrics
from keras import initializers
from keras import utils
from keras.utils.generic_utils import get_custom_objects
# from keras import losses 
# from keras.callbacks import TensorBoard
# from keras.callbacks import LambdaCallback
#from keras.callbacks import ModelCheckpoint
#from keras.callbacks import LearningRateScheduler
#from keras.callbacks import EarlyStopping

#from tensorflow.keras.models import Model
#from tensorflow.keras.callbacks import  ModelCheckpoint

#import cv2
#import glob


#import time 
#from sympy import *

import sys
import matplotlib.pyplot as plt



N=40; dx=1./N
def f(x): return 1-np.cos(6.28*x)
def f_2(x): return 6.28*6.28*np.cos(6.28*x) #
def init_W0(shape, dtype=None):
    W=np.array([np.ones(N)]); return K.constant(W)
def init_b0(shape, dtype=None): return K.constant(-np.linspace(0,1,N))
def init_W1(shape, dtype=None):
    array3=np.zeros(N)
    for i in range(0,N): array3[i]=f_2(i*dx)*dx
    return K.constant(np.transpose(np.array([array3])))

model = Sequential()
model.add(Dense(N, input_dim=1,name="lay_in",kernel_initializer=init_W0,
                use_bias=True, bias_initializer=init_b0, activation='relu'))
model.add(Dense(1,name="lay_out",kernel_initializer=init_W1,
                use_bias=False,activation='linear'))

x_p=np.linspace(0,1,N); y_p=model.predict(x_p); 


plt.figure(dpi=600)
plt.plot(x_p,y_p,'-+',label='N=40')
plt.xlabel('x') 
plt.ylabel('f(x)') 
plt.legend(loc="upper left")
plt.show()






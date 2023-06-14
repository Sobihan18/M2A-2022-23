#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 12:05:15 2023

@author: despres
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import datetime as dt

import numpy as np
import tensorflow
import tensorflow.keras as keras
from keras import backend as K

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam,SGD
from keras.layers import Input, Dense, Dropout, Reshape,Flatten,BatchNormalization, Activation,ReLU
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
# from keras import metrics
from keras import initializers
from tensorflow.keras import initializers,activations

from keras.layers import LeakyReLU



from keras import utils
from keras.utils.generic_utils import get_custom_objects
# from keras import losses 
# from keras.callbacks import TensorBoard
# from keras.callbacks import LambdaCallback
#from keras.callbacks import ModelCheckpoint
#from keras.callbacks import LearningRateScheduler
#from keras.callbacks import EarlyStopping

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation
#from tensorflow.keras.models import Model
#from tensorflow.keras.callbacks import  ModelCheckpoint
from keras.callbacks import LearningRateScheduler

import random 
import matplotlib.pyplot as plt



N=20 # nombre Fourier modes
N_Data=20000 # Data

M=100
dx=1./M
x=np.linspace(0,1,M+1)
y=np.zeros(M+1)
f=np.zeros(M+1)
pi=3.14159265358979323846 

mul=20

string_file='%5.4f,'
string_file_int='%2.d,'
string_line_end=' \n'


strstr_train='../DATA/TRAIN_lap.txt'
fichier_train = open(strstr_train,'w')
strstr_test='../DATA/TEST_lap.txt'
fichier_test  = open(strstr_test,'w')
strstr_format='../DATA/FORMAT_lap.txt'
fichier_format = open(strstr_format,'w')

for l in range(0,N_Data):
    print ("data=",l)

    for k in range(0,M+1):
            y[k]=0.
            
    for i in range(1,N):

        amp=10*random.uniform(-1,1)/(((i+1)*pi))
        #print ("   i=",i,", amp=",amp)

        for k in range(0,M+1):
            y[k]=y[k]+amp*np.sin(pi*i*k*dx)
                        
    for k in range(1,M):
            f[k]=-(y[k+1]-2*y[k]+y[k-1])/(dx**2)+mul*y[k]**3
            
            
                
    if (np.random.uniform(0., 1.)>0.2):
        for k in range(1,M):
            fichier_train.write(string_file % f[k])
        for k in range(1,M):
            fichier_train.write(string_file % y[k])
        fichier_train.write('%5.4f' % -1000000) 
        fichier_train.write(string_line_end)    
    else:
        for k in range(1,M):
            fichier_test.write(string_file % f[k])
        for k in range(1,M):
            fichier_test.write(string_file % y[k])
        fichier_test.write('%5.4f' % -1000000) 
        fichier_test.write(string_line_end) 

zz=M-1.
fichier_format.write(string_file_int %  zz) 
fichier_format.close()
fichier_train.close()
fichier_test.close()


plt.plot(x,f,'-',label='f(x)')
plt.legend(loc="upper left")
plt.show()

plt.plot(x,y,'-',label='y(x)')
plt.legend(loc="upper left")
plt.show()



print("Nombrede modes= ",N-1)
print("This is the end")

"""
Training for the classification of curves
# modificatin of a CNN used for MNIST 
B. Despres
01/11/2021
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
from keras.layers import Dense,Input
from keras.optimizers import Adam
from keras.layers import Input, Dense, Dropout, Reshape,Flatten,BatchNormalization, Activation,ReLU
from keras.layers import Conv2D, MaxPooling2D,Conv1D,MaxPooling1D,Reshape
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

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation
#from tensorflow.keras.models import Model
#from tensorflow.keras.callbacks import  ModelCheckpoint
from keras.callbacks import LearningRateScheduler

#import cv2
#import glob


#import time 
#from sympy import *

import numpy as np


import sys
import matplotlib.pyplot as plt






print("\n------------------------------------------\n")

start = dt.datetime.now()

SEED = 2020; np.random.seed(SEED); tensorflow.compat.v1.random.set_random_seed(SEED)

# Loading datasets
DATA_DIR = "../DATA/"


#nombre_labels=np.loadtxt(DATA_DIR + "FORMAT_tak.txt", dtype=np.int32)

strstr_train='../DATA/FORMAT_lap.txt'
fichier_format = open(strstr_train,'r')
infos = fichier_format.readline(-1).split(",")
nombre_labels=int(infos[0])
nombre_points=nombre_labels

print("nombre_labels=",nombre_labels)
print("nombre_points=",nombre_points)

fichier_format.close()



train_data = np.loadtxt(DATA_DIR + "TRAIN_lap.txt", dtype=np.float32, delimiter=',')
np.random.shuffle(train_data)
x_train = train_data[:, 0:nombre_labels]
y_train = train_data[:, nombre_labels:nombre_labels+nombre_points]

print ("Taille x_train=",x_train.shape)
print ("Taille y_train=",y_train.shape)

#sys.exit(0)


test_data = np.loadtxt(DATA_DIR + "TEST_lap.txt", dtype=np.float32, delimiter=',')
np.random.shuffle(test_data)
x_test = test_data[:, 0:nombre_labels]
y_test = test_data[:, nombre_labels:nombre_labels+nombre_points]
print ("Taille x_test=",x_test.shape)
print ("Taille y_test=",y_test.shape)


#sys.exit(0)





#--- Fonction activatin T-ReLU -------#


def T_relu(x):
    return K.relu(x, max_value=1)

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        
class VallLossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('acc'))
        
class LossEpochHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        
class LearnrateEpochHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('lr'))
        

def scheduler(epoch, lr):
    return 0.001#/(epoch+1)**0.25    

history = LossHistory()
historyval = VallLossHistory()
historyepoch = LossEpochHistory()
historylearnrate=LearnrateEpochHistory()
lrate = LearningRateScheduler(scheduler,verbose=0)



#------ Defining model -------#

input_shape=(nombre_points)

#x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
#x_test =  x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

print ("Taille_reshape x_train=",x_train.shape)


model = Sequential()
# model.add(Conv1D(30,2, input_shape=input_shape,
#                 name="couche_entree",
#                 kernel_initializer='random_uniform',
#                 use_bias=False, bias_initializer='random_uniform', 
#                 activation='linear'))


# model.add(MaxPooling1D())

# model.add(Conv1D(10,2, 
#                 kernel_initializer='random_uniform',
#                 use_bias=False, bias_initializer='random_uniform', 
#                 activation='linear'))

# model.add(MaxPooling1D())


# model.add(Flatten())

# model.add(Conv1D(filters=99, kernel_size=(3),
#                 kernel_initializer='random_uniform',
#                 use_bias=False, bias_initializer='random_uniform', 
#                 activation='linear'))




model.add(Dense(30,name="couche_hidden_1",
                input_dim=nombre_points,
                kernel_initializer='random_uniform',
                use_bias=False, bias_initializer='random_uniform', 
                activation='linear'))


model.add(Dense(30,name="couche_hidden_2",
                input_dim=nombre_points,
                kernel_initializer=initializers.Constant(value=0.1),
                use_bias=False, bias_initializer='random_uniform', 
                activation='relu'))

model.add(Dense(20,name="couche_hidden_3",
                input_dim=nombre_points,
                kernel_initializer='random_uniform',
                use_bias=False, bias_initializer='random_uniform', 
                activation='relu'))

model.add(Dense(20,name="couche_hidden_4",
                input_dim=nombre_points,
                kernel_initializer='random_uniform',
                use_bias=False, bias_initializer='random_uniform', 
                activation='relu'))

model.add(Dense(20,name="couche_hidden_5",
                input_dim=nombre_points,
                kernel_initializer='random_uniform',
                use_bias=False, bias_initializer='random_uniform', 
                activation='relu'))


# model.add(Dense(3,name="couche_hidden_2",
#                 kernel_initializer='random_uniform',
#                 use_bias=True, bias_initializer='random_uniform', 
#                 activation='relu'))
# model.add(Dense(2,name="couche_hidden_3",
#                 kernel_initializer='random_uniform',
#                 use_bias=True, bias_initializer='random_uniform', 
#                 activation='relu'))


model.add(Dense(nombre_labels,name="couche_sortie",
                kernel_initializer='random_uniform',
                use_bias=False, bias_initializer='random_uniform', 
                activation='linear'))

model.compile(loss='mse',optimizer="Adam")

model.summary()

#sys.exit(0)


score_ini = model.evaluate(x_test, y_test, verbose=1)
print('Score_ini = ',score_ini)

print("Fin initialisation")
print(" ")



# Learning
model.fit(x_train, y_train,
		  batch_size=128,
		  epochs=200,
		  verbose=1, validation_data=(x_test, y_test)
		  ) 

print(" ")


score = model.evaluate(x_test, y_test)
print("Score_fin = ",score)

y_predict=model.predict(x_test,verbose=1)



xxx=x_test[[1]]
yyy=y_test[[1]]
zzz=model.predict(xxx)
#sys.exit(0)
    
xxx_2=np.linspace(0,1,nombre_points)
xxx_4=np.linspace(0,1,nombre_points)
yyy_2=np.linspace(0,1,nombre_points)
zzz_2=np.linspace(0,1,nombre_points)
www_2=np.linspace(0,1,nombre_points)
for k in range(0,nombre_points):
        xxx_4[k]=xxx[0][k]
        yyy_2[k]=yyy[0][k]
        zzz_2[k]=zzz[0][k]
    
   
#y_predict=model.predict(x_p)    
plt.plot(xxx_2,xxx_4)
plt.show()
plt.plot(xxx_2,yyy_2)
plt.show()
plt.plot(xxx_2,zzz_2)
plt.show()

dx=1/nombre_points
pi=3.14159265358979323846 

for k in range(0,nombre_points):
    xxx[0][k]=20# 10*6*k*dx#np.sin(pi*k*dx)#(k*dx)*(1-k*dx)

www=model.predict(xxx)
for k in range(0,nombre_points):
        www_2[k]=www[0][k]
        
plt.plot(xxx_2,www_2)
plt.show()

#------------  Sortie  -----------#
string_file='%5.8f %5.8f %5.8f %5.8f'
string_file=string_file+' \n'

strstr ='./sortie.plot'
fichier = open(strstr,'w')

#for i in range(0,n):
#    fichier.write(string_file %  (x_p[i],z_p[i],y_predict_ini[i],y_predict[i]) )

fichier.close()
#------------  Sortie  -----------#
    
    

end = dt.datetime.now()
print("Training duration: {} seconds".format(end - start))


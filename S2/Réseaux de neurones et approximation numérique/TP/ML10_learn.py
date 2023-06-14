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
from keras.optimizers import Adam
from keras.layers import Input, Dense, Dropout, Reshape,Flatten,BatchNormalization, Activation,ReLU
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation
#from tensorflow.keras.models import Model
#from tensorflow.keras.callbacks import  ModelCheckpoint
from keras.callbacks import LearningRateScheduler


#import cv2
#import glob


#import time 
#from sympy import *



import sys
import matplotlib.pyplot as plt



print("\n------------------------------------------\n")




N_interp=3 # Ncell



print("\n------------------------------------------\n")

start = dt.datetime.now()

SEED = 2020; np.random.seed(SEED); tensorflow.compat.v1.random.set_random_seed(SEED)

# Loading datasets
DATA_DIR = "../DATA/"
train_data = np.loadtxt(DATA_DIR + "TRAIN_tak.txt", dtype=np.float32, delimiter=',')
np.random.shuffle(train_data)
x_train = train_data[:, 1:N_interp+2]
y_train = train_data[:, 0:1]

print ("Taille x_train=",x_train.shape)
print ("Taille y_train=",y_train.shape)

test_data = np.loadtxt(DATA_DIR + "TEST_tak.txt", dtype=np.float32, delimiter=',')
#np.random.shuffle(test_data)
x_test = test_data[:, 1:N_interp+2]
y_test = test_data[:, 0:1]
print ("Taille x_test=",x_test.shape)
print ("Taille y_test=",y_test.shape)


print("fin lecture")
#sys.exit(0)

#------ initialiseur couches ReLU ---#
#------ initialiseur couches ReLU ---#

fac=3
depth=5
nombre_points=50
dx=1./nombre_points
array_numpy=np.ones(nombre_points)





############################################################
############################################################
############################################################

#
#  callbacks
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
    return 0.001/(epoch+1)**0.25     
   
lrate = LearningRateScheduler(scheduler,verbose=0)

lambda_mse = 0.
def my_loss (y_true, y_pred):
    # mse
    mse_loss = (2*K.mean(K.square(K.abs(K.abs(K.abs(y_true - y_pred))))))**1.    
    msq_loss = K.mean(K.abs(y_true - y_pred))
    ae_loss = K.mean(K.abs(y_true - y_pred))**1
    return mse_loss + msq_loss+lambda_mse * ae_loss


#regularizer = K.nn.l2_loss(weights)
        

history = LossHistory()
historyval = VallLossHistory()
historyepoch = LossEpochHistory()
historylearnrate=LearnrateEpochHistory()


#------ Defining model -------#
model = Sequential()



model.add(Dense(nombre_points, input_dim=N_interp+1,name="couche_entree",
                use_bias=True, bias_initializer='random_uniform',
                activation='relu'))

model.add(Dense(nombre_points,name="couche_h1",
                use_bias=True, bias_initializer='random_uniform',
                activation='relu'))

model.add(Dense(nombre_points,name="couche_h2",
                #kernel_initializer='random_uniform',
                #kernel_regularizer=keras.regularizers.l1(l=0.01),
                use_bias=True, bias_initializer='random_uniform',
                #bias_regularizer=keras.regularizers.l1(l=0.01),
                activation='relu'))

model.add(Dense(nombre_points,name="couche_h3",
                #kernel_initializer='random_uniform',
                #kernel_regularizer=keras.regularizers.l1(l=0.01),
                use_bias=True, bias_initializer='random_uniform',
                #bias_regularizer=keras.regularizers.l1(l=0.01),
                activation='relu'))

model.add(Dense(nombre_points,name="couche_h4",
                #kernel_initializer='random_uniform',
                #kernel_regularizer=keras.regularizers.l1(l=0.01),
                use_bias=True, bias_initializer='random_uniform',
                #bias_regularizer=keras.regularizers.l1(l=0.01),
                activation='relu'))

model.add(Dense(nombre_points,name="couche_h5",
                #kernel_initializer='random_uniform',
                #kernel_regularizer=keras.regularizers.l1(l=0.01),
                use_bias=True, bias_initializer='random_uniform',
                #bias_regularizer=keras.regularizers.l1(l=0.01),
                activation='relu'))

model.add(Dense(1,name="couche_sortie",
                #kernel_initializer='random_uniform',
                #kernel_regularizer=keras.regularizers.l1(l=0.01),
                use_bias=True, bias_initializer='random_uniform',
                activation='linear'))





model.compile(loss='mse', optimizer=Adam())

W_ini=model.get_weights()




rmse = np.sqrt(model.evaluate(x_test, y_test, verbose=False))
print('Init l2-error: ',format(rmse))
print("Fin initialisation")



# Learning
model.fit(x_train, y_train,
		  batch_size=800,
		  epochs=100,
          callbacks=[history,historyepoch,historyval,lrate,historylearnrate],
		  verbose=1, validation_data=(x_test, y_test)
		  ) 

W_fin=model.get_weights()


rmse = np.sqrt(model.evaluate(x_test, y_test, verbose=False))

y_prod=model.predict(x_test,verbose=1)    

print(x_test)
print([[1,2,3,4]])
    
taille=N_interp+1
x_mpl=np.array([np.array(taille*[np.float32(0.)])])
print(x_mpl)

x_mpl[0][0]=-0.5
x_mpl[0][1]=0.3
x_mpl[0][2]=-0.
x_mpl[0][3]=-0.3

#np.array([[1,2,3,4]])


y_sortie=model.predict(x_mpl,verbose=1)    

print("x=",x_mpl,", y=",y_sortie)

    

end = dt.datetime.now()
print("Training duration: {} seconds".format(end - start))

model.save('../DATA/my_model')


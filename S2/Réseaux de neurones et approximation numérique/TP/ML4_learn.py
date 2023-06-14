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
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Input, Dense, Dropout, Reshape,Flatten,BatchNormalization, Activation,ReLU
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

start = dt.datetime.now()

SEED = 2020; np.random.seed(SEED); tensorflow.compat.v1.random.set_random_seed(SEED)

# Loading datasets
DATA_DIR = "../DATA/"


#nombre_labels=np.loadtxt(DATA_DIR + "FORMAT_tak.txt", dtype=np.int32)

strstr_train='../DATA/FORMAT_curves.txt'
fichier_format = open(strstr_train,'r')
infos = fichier_format.readline(-1).split(",")
nombre_labels=int(infos[0])
nombre_points=int(infos[1])

print("nombre_labels=",nombre_labels)
print("nombre_points=",nombre_points)

fichier_format.close()




train_data = np.loadtxt(DATA_DIR + "TRAIN_curves.txt", dtype=np.float32, delimiter=',')
np.random.shuffle(train_data)
x_train = train_data[:, nombre_labels:nombre_labels+nombre_points]
y_train = train_data[:, 0:nombre_labels]

print ("Taille x_train=",x_train.shape)
print ("Taille y_train=",y_train.shape)



test_data = np.loadtxt(DATA_DIR + "TEST_curves.txt", dtype=np.float32, delimiter=',')
np.random.shuffle(test_data)
x_test = test_data[:, nombre_labels:nombre_labels+nombre_points]
y_test = test_data[:, 0:nombre_labels]
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
model = Sequential()
model.add(Dense(20, input_dim=nombre_points,name="couche_entree",
                kernel_initializer='random_uniform',
                use_bias=True, bias_initializer='random_uniform', 
                activation='relu'))

model.add(Dense(50,name="couche_hidden_1",
                kernel_initializer='random_uniform',
                use_bias=True, bias_initializer='random_uniform', 
                activation='relu'))
model.add(Dense(30,name="couche_hidden_2",
                kernel_initializer='random_uniform',
                use_bias=True, bias_initializer='random_uniform', 
                activation='relu'))
model.add(Dense(20,name="couche_hidden_3",
                kernel_initializer='random_uniform',
                use_bias=True, bias_initializer='random_uniform', 
                activation='relu'))


model.add(Dense(nombre_labels,name="couche_sortie",
                kernel_initializer='random_uniform',
                use_bias=True, bias_initializer='random_uniform', 
                activation='softmax'))

model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'],
              optimizer=Adam(beta_1=0.9,beta_2=0.999))



score_ini = model.evaluate(x_test, y_test, verbose=1)
print('Score_ini = ',score_ini)

print("Fin initialisation")
print(" ")



# Learning
model.fit(x_train, y_train,
		  batch_size=100,
		  epochs=200,
          #callbacks=[history,historyepoch,historyval,lrate,historylearnrate],
		  verbose=1, validation_data=(x_test, y_test)
		  ) 

print(" ")


score = model.evaluate(x_test, y_test, verbose=1)
print("Score_fin = ",score)

y_predict=model.predict(x_test,verbose=1)





for k in range(0,2): 
    print ('y_predict',y_predict[[k]],': y_vrai   ',y_test[[k]]) 
    
sys.exit(0)
    
    
   
y_predict=model.predict(x_p)    
plt.plot(x_p,y_predict)


#------------  Sortie  -----------#
string_file='%5.8f %5.8f %5.8f %5.8f'
string_file=string_file+' \n'

strstr ='./sortie.plot'
fichier = open(strstr,'w')

for i in range(0,n):
    fichier.write(string_file %  (x_p[i],z_p[i],y_predict_ini[i],y_predict[i]) )

fichier.close()
#------------  Sortie  -----------#
    
    

end = dt.datetime.now()
print("Training duration: {} seconds".format(end - start))


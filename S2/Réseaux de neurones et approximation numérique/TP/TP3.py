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


#import cv2
#import glob


#import time 
#from sympy import *



import sys
import matplotlib.pyplot as plt

import ast



print("\n------------------------------------------\n")

def g1(x):
    return 2*x*(x<0.5) + (2-2*x)*(x>=0.5)
def g2(x):
    return g1(g1(x))
def g3(x):
    return g1(g2(x))
def g4(x):
    return g1(g3(x))
def g5(x):
    return g1(g4(x))
def g6(x):
    return g1(g5(x))
def g7(x):
    return g1(g6(x))
def g8(x):
    return g1(g7(x))
def g9(x):
    return g1(g8(x))

a=4
def Tak(x):
    b=1*( g1(x)/a+g2(x)/a**2+g3(x)/a**3
       +g4(x)/a**4+g5(x)/a**5+g6(x)/a**6 
       +g7(x)/a**7+g8(x)/a**8+g9(x)/a**9 ) #+20*x**5*(1-x)
    b=x*x
    b=x-x*x*x*x*x*x*x
    b=x-x*x
    return b




n=100
x_p=np.linspace(0,1,n)
y_p=g2(x_p)
z_p=Tak(x_p)

plt.figure(dpi=600)


plt.plot(x_p,z_p,label='fobj(x)')

print("Fin partie 1")
#sys.exit(0)




print("\n------------------------------------------\n")
print("\n          Chargement donnees n")
print("\n------------------------------------------\n")


start = dt.datetime.now()

SEED = 2021; np.random.seed(SEED); tensorflow.compat.v1.random.set_random_seed(SEED)

# Loading datasets
DATA_DIR = "../DATA/"
train_data = np.loadtxt(DATA_DIR + "TRAIN_tak.txt", dtype=np.float32, delimiter=',')
np.random.shuffle(train_data)
x_train = train_data[:, 0:1]
y_train = train_data[:, 1:2]



print ("Taille x_train=",x_train.shape)
print ("Taille y_train=",y_train.shape)

test_data = np.loadtxt(DATA_DIR + "TEST_tak.txt", dtype=np.float32, delimiter=',')
np.random.shuffle(test_data)
x_test = test_data[:, 0:1]
y_test = test_data[:, 1:2]


print ("Taille x_test=",x_test.shape)
print ("Taille y_test=",y_test.shape)


#------ initialiseur couches ReLU ---#
#------ initialiseur couches ReLU ---#

fac=4
depth=4


def init_W0(shape, dtype=None):
    W=np.array([[1,1,0 ]])
    return K.constant(W)
def init_b0(shape, dtype=None):
    b=np.array([0,-1./2.,0 ])
    return K.constant(b)

        


def init_W1(shape, dtype=None):
    W=np.array([[2,2,2/fac],[-4,-4,-4/fac],[0,0,1]])
    return K.constant(W)   
def init_b1(shape, dtype=None):
    b=np.array([0,-1./2.,0 ])
    return K.constant(b)

def init_W2(shape, dtype=None):
    W=np.array([[2,2,2/fac**2],[-4,-4,-4/fac**2],[0,0,1]])
    return K.constant(W)   
def init_b2(shape, dtype=None):
    b=np.array([0,-1./2.,0 ])
    return K.constant(b)

def init_W3(shape, dtype=None):
    W=np.array([[2,2,2/fac**3],[-4,-4,-4/fac**3],[0,0,1]])
    return K.constant(W)   
def init_b3(shape, dtype=None):
    b=np.array([0,-1./2.,0 ])
    return K.constant(b)

def init_W4(shape, dtype=None):
    W=np.array([[2,2,2/fac**4],[-4,-4,-4/fac**4],[0,0,1]])
    return K.constant(W)   
def init_b4(shape, dtype=None):
    b=np.array([0,-1./2.,0 ])
    return K.constant(b)

def init_W5(shape, dtype=None):
    W=np.array([[2,2,2/fac**5],[-4,-4,-4/fac**5],[0,0,1]])
    return K.constant(W)   
def init_b5(shape, dtype=None):
    b=np.array([0,-1./2.,0 ])
    return K.constant(b)

def init_W_sortie(shape, dtype=None):
    W=np.array([[2/fac**depth],[-4/fac**depth],[1]])
    return K.constant(W)
def init_b_sortie(shape, dtype=None):
    b=np.array([0.5])
    return K.constant(b)

def init_W_spec(shape, dtype=None):
    W=np.array([[3,3,3],[-6,-6,-6],[6,6,6]])
    KKK=K.constant(W)
    return KKK   



#------ Defining model -------#
model = Sequential()
model.add(Dense(3, input_dim=1,name="couche_entree",
                kernel_initializer=init_W0,
                use_bias=True, bias_initializer=init_b0,
                activation='relu'))

model.add(Dense(3,name="couche_hidden_1",
                kernel_initializer=init_W1,
                use_bias=True, bias_initializer=init_b1,
                activation='relu'))



model.add(Dense(3,name="couche_hidden_2",
                kernel_initializer=init_W2,
                use_bias=True, bias_initializer=init_b2, 
                activation='relu'))

model.add(Dense(3,name="couche_hidden_3",
                kernel_initializer=init_W3,
                use_bias=True, bias_initializer=init_b3, 
                activation='relu'))


model.add(Dense(1,name="couche_sortie",
                kernel_initializer=init_W_sortie,
                use_bias=False, bias_initializer=init_b_sortie, 
                activation='linear'))

Wini=model.get_weights()


model.compile(loss='mse', optimizer=Adam())


model.summary()

y_predict_ini=model.predict(x_p)    
plt.plot(x_p,y_predict_ini,'-',label='f3(x)')
plt.xlabel('x') 
plt.ylabel('') 
plt.legend(loc="upper left")


print("Fin initialisation 2")
#sys.exit(0)



# Learning
toto=model.fit(x_train, y_train,
		  batch_size=256,
		  epochs=100,
		  verbose=1, validation_data=(x_test, y_test)
		  ) 

Wfin=model.get_weights()


rmse = np.sqrt(model.evaluate(x_train, y_train, verbose=False))
print('Test l2-error: {:.6f}'.format(rmse))


   
y_predict=model.predict(x_p)    
diff_y=np.linspace(0,1,n)
for i in range(0,n):
        diff_y[i]=z_p[i]-y_predict[i]

#plt.plot(x_p,diff_y,'-')

plt.plot(x_p,y_predict,'-',label='f(x)')

plt.show()

#------------  Sortie  -----------#
string_file='%5.8f %5.8f %5.8f %5.8f'
string_file=string_file+' \n'

strstr ='./sortie.plot'
fichier = open(strstr,'w')
for i in range(0,n):
    fichier.write(string_file %  
                  (x_p[i],z_p[i],y_predict_ini[i],y_predict[i]) )
fichier.close()
#------------  Sortie  -----------#
    
string_file='%5.8f'
string_file=string_file+' \n'
strstr ='./historyepoch.txt'
fichier = open(strstr,'w')
for i in range(0,len(historyepoch.losses)):
        fichier.write(string_file %  (historyepoch.losses[i]))
        #print (i,history.losses[i])
fichier.close() 

strstr ='./historyall.txt'
fichier = open(strstr,'w')
for i in range(0,len(history.losses)):
        fichier.write(string_file %  (history.losses[i]))
        #print (i,history.losses[i])
fichier.close() 


strstr ='./historylearnrate.txt'
fichier = open(strstr,'w')
for i in range(0,len(historylearnrate.losses)):
        fichier.write(string_file %  (historylearnrate.losses[i]))
fichier.close() 

end = dt.datetime.now()
print("Training duration: {} seconds".format(end - start))


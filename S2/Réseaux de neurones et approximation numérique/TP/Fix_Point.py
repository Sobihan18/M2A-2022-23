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

#import cv2
#import glob


#import time 
#from sympy import *



import sys
import matplotlib.pyplot as plt

import ast



def fobj(x):
    return x*x*x

n=100
x_p=np.linspace(0,1,n)
y_p=fobj(x_p)


plt.figure(dpi=600)

plt.plot(x_p,y_p,label="obj")

print("Fin partie 1")
#sys.exit(0)




print("\n------------------------------------------\n")

start = dt.datetime.now()

SEED = 2020; np.random.seed(SEED); tensorflow.compat.v1.random.set_random_seed(SEED)



#------ initialiseur couches TReLU ---#
#------ initialiseur couches TReLU ---#

fac=4.
depth=4


def init_W0_T(shape, dtype=None):
    W=np.array([[1,2,-2,2 ]])
    return K.constant(W)

def init_b0_T(shape, dtype=None):
    b=np.array([0,0,2, -1 ])
    return K.constant(b)


def init_T_W_e0(shape, dtype=None):
    W=np.array([[3./2.]])
    return K.constant(W)

def init_T_b_e0(shape, dtype=None):
    b=np.array([-3./4. ])
    return K.constant(b)

def init_T_W_e00(shape, dtype=None):
    W=np.array([[1]])
    return K.constant(W)

def init_T_b_e00(shape, dtype=None):
    b=np.array([-1./8.])
    return K.constant(b)



# def init_W2_T(shape, dtype=None):
#     W=np.array([[2,-2,1/fac**1],[2,-2,1/fac**1],[0,0,1]])
#     return K.constant(W)   
# def init_b2_T(shape, dtype=None):
#     b=np.array([-2,4,-1/fac**1 ])
#     return K.constant(b)

# def init_W3_T(shape, dtype=None):
#     W=np.array([[2,2,1/fac**2],[-4,-4,-4/fac**2],[0,0,1]])
#     return K.constant(W)   
# def init_b3_T(shape, dtype=None):
#     b=np.array([0,-1./2.,0 ])
#     return K.constant(b)

# def init_W4_T(shape, dtype=None):
#     W=np.array([[2,2,1/fac**3],[-4,-4,-4/fac**3],[0,0,1]])
#     return K.constant(W)   
# def init_b4_T(shape, dtype=None):
#     b=np.array([0,-1./2.,0 ])
#     return K.constant(b)

# def init_W5_T(shape, dtype=None):
#     W=np.array([[2,2,2/fac**4],[-4,-4,-4/fac**4],[0,0,1]])
#     return K.constant(W)   
# def init_b5_T(shape, dtype=None):
#     b=np.array([0,-1./2.,0 ])
#     return K.constant(b)

def init_W_sortie_T(shape, dtype=None):
    W=np.array([[1],[1./8.],[1./8.],[1./4.]])
    #W=np.array([[1],[0],[0],[0]])
    return K.constant(W)
def init_b_sortie_T(shape, dtype=None):
    b=np.array([0])
    return K.constant(b)


def init_W_sortie_T_16(shape, dtype=None):
    W=np.array([[1],[1./8.],[1./8.],[1./4.],
                [1*0],[1./8.*1./8.],[1./8.*1./8.],[1./4.*1./8.],
                [1*0.],[1./8.*1./8.],[1./8.*1./8.],[1./4.*1./8.],
                [1*0.],[1./8.*1./4.],[1./8.*1./4.],[1./4.*1./4.],
                ])
    #W=np.array([[1],[0],[0],[0]])
    return K.constant(W)
def init_b_sortie_T_16(shape, dtype=None):
    b=np.array([0])
    return K.constant(b)

def init_W_sortie_T_64(shape, dtype=None):
    a=1/8.
    b=1/4.
    L=[1,a,a,b]
    S=2*L
    C=[[1,a,a,b],[0,a*a,a*a,b*a],[0,a*a,a*a,b*a],[0,a*b,a*b,b*b]]
    
    D=[[0,0,0,0],[0,a*a*a,a*a*a,b*a*a],[0,a*a*a,a*a*a,b*a*a],[0,a*b*a,a*b*a,b*b*a]]
    E=[[0,0,0,0],[0,a*a*a,a*a*a,b*a*a],[0,a*a*a,a*a*a,b*a*a],[0,a*b*a,a*b*a,b*b*a]]
    F=[[0,0,0,0],[0,a*a*b,a*a*b,b*a*b],[0,a*a*b,a*a*b,b*a*b],[0,a*b*b,a*b*b,b*b*b]]

    W=[C,D,E,F]
    Z=np.array(W)
    print ("Z",Z)
    X=[Z.flatten()]
    Y=np.transpose(X)
    print ("Y",Y)
    return K.constant(Y)



#--- Fonction activatin T-ReLU -------#


def T_relu(x):
    return K.relu(x, max_value=1)


#------ Defining model -------#
model = Sequential()
model.add(Dense(4, input_dim=1,
                kernel_initializer=init_W0_T,
                use_bias=True, bias_initializer=init_b0_T, 
                activation=T_relu))

model.add(Reshape((4,1)))
model.add(Dense(4, input_dim=1,
                kernel_initializer=init_W0_T,
                use_bias=True, bias_initializer=init_b0_T, 
                activation=T_relu))
model.add(Flatten())
model.add(Reshape((16,1)))

model.add(Dense(4, input_dim=1,
                kernel_initializer=init_W0_T,
                use_bias=True, bias_initializer=init_b0_T, 
                activation=T_relu))
model.add(Flatten())
model.add(Reshape((64,1)))


model.add(Dense(1, input_dim=1,
                kernel_initializer=init_T_W_e0,
                use_bias=True, bias_initializer=init_T_b_e0, 
                activation=T_relu))

model.add(Dense(1, input_dim=1,
                kernel_initializer=init_T_W_e00,
                use_bias=True, bias_initializer=init_T_b_e00, 
                activation='linear'))

model.add(Flatten())


# model.add(Dense(3,name="couche_hidden_1",
#                 kernel_initializer=init_W1,
#                 use_bias=True, bias_initializer=init_b0, 
#                 activation='relu'))
# model.add(Dense(3,name="couche_hidden_2",
#                 kernel_initializer=init_W2,
#                 use_bias=True, bias_initializer=init_b0, 
#                 activation='relu'))
# model.add(Dense(3,name="couche_hidden_3",
#                 kernel_initializer=init_W3,
#                 use_bias=True, bias_initializer=init_b3, 
#                 activation='relu'))
# model.add(Dense(3,name="couche_hidden_4",
#                 kernel_initializer=init_W4,
#                 use_bias=True, bias_initializer=init_b4, 
#                 activation='relu'))
# model.add(Dense(3,name="couche_hidden_5",
#                 kernel_initializer=init_W5,
#                 use_bias=True, bias_initializer=init_b5, 
#                 activation='relu'))

model.add(Dense(1,
               kernel_initializer=init_W_sortie_T_64,
               use_bias=False, activation='linear'))



#model.compile(loss='mse', optimizer=Adam())

model.summary()

#sys.exit(0)


y_predict_ini=model.predict(x_p)    
plt.plot(x_p,y_predict_ini,'-',label='f64(x)')
plt.xlabel('x') 
plt.ylabel('y') 
plt.legend(loc="upper left")
plt.show()


norme=0.
for i in range(0,n): 
    norme=norme+ (y_p[i]-y_predict_ini[i])**2
norme=np.sqrt(norme/n)
print ("Erreur L2=",norme)

print("Fin initialisation 2")
sys.exit(0)



# Learning
# model.fit(x_train, y_train,
# 		  batch_size=100,
# 		  epochs=100,
# 		  verbose=1, validation_data=(x_test, y_test)
# 		  ) 

rmse = np.sqrt(model.evaluate(x_test, y_test, verbose=False))
print('Test l2-error: {:.6f}'.format(rmse))


#print (' ')
#y_essai=model.predict(x_test,verbose=1)    
#for i5 in range(0,20): 
#    print ('1  : y_predict',y_essai[[i5]],': y_vrai   ',y_test[[i5]]) #,': x  ',x_test[[i5]])
    
   
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


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


from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation
from keras.callbacks import LearningRateScheduler
from tensorflow.keras import layers



import sys
import matplotlib.pyplot as plt

import ast





n=1000
x_p=np.linspace(0,1,n)






#------ initialiseur couches ReLU ---#
#------ initialiseur couches ReLU ---#

fac=4
depth=1


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

def init_W6(shape, dtype=None):
    W=np.array([[2,2,2/fac**6],[-4,-4,-4/fac**6],[0,0,1]])
    return K.constant(W)   
def init_b6(shape, dtype=None):
    b=np.array([0,-1./2.,0 ])
    return K.constant(b)

def init_W_sortie(shape, dtype=None):
    W=np.array([[2/fac**depth],[-4/fac**depth],[1]])
    return K.constant(W)
def init_b_sortie(shape, dtype=None):
    b=np.array([0.5])
    return K.constant(b)

def init_Z1(shape, dtype=None):
    W=np.array([[-1],[1]])
    return K.constant(W) 

def init_Z2(shape, dtype=None):
    W=np.array([[1./2.],[1./2.]])
    return K.constant(W) 

def init_Z3(shape, dtype=None):
    W=np.array([[-1./2.],[-1./2.]])
    return K.constant(W) 


def init_Z4(shape, dtype=None):
    W=np.array([[1.],[-1.]])
    return K.constant(W) 


inputs = keras.Input(shape=(1),name="inputs")





print("Debut construction du multiplicateur")

####  x-> x^2   ####
model2=layers.Dense(3, activation='relu',
                kernel_initializer=init_W0,
                use_bias=True, bias_initializer=init_b0)(inputs)

# model3=layers.Dense(3, activation='relu',
#                 kernel_initializer=init_W1,
#                 use_bias=True, bias_initializer=init_b1)(model2)

# model4=layers.Dense(3, activation='relu',
#                 kernel_initializer=init_W2,
#                 use_bias=True, bias_initializer=init_b2)(model3)

# model5=layers.Dense(3, activation='relu',
#                 kernel_initializer=init_W3,
#                 use_bias=True, bias_initializer=init_b3)(model4)

# model6=layers.Dense(3, activation='relu',
#                 kernel_initializer=init_W4,
#                 use_bias=True, bias_initializer=init_b4)(model5)

# model7=layers.Dense(3, activation='relu',
#                 kernel_initializer=init_W5,
#                 use_bias=True, bias_initializer=init_b5)(model6)

model=layers.Dense(1, activation='relu',
                kernel_initializer=init_W_sortie,
                use_bias=False)(model2)




##  x->x  ###
initializer = tensorflow.keras.initializers.Constant(1.)
identity=layers.Dense(1, activation='linear',
                          kernel_initializer=initializer,
                          use_bias=False)(inputs)
### x-> (x-x^2,x)
Lign1=layers.concatenate([model, identity])
### x-> x^2
Lign2=layers.Dense(1, activation='linear',
                kernel_initializer=init_Z1,
                use_bias=False)(Lign1)
### x-> x^2
model_comp=keras.Model(inputs=inputs,outputs=Lign2)

### x-> (x^2,x)
Lign3=layers.concatenate([Lign2, identity])
### x-> (x^2+x)/2 
Lign4=layers.Dense(1, activation='linear',
                kernel_initializer=init_Z2,
                use_bias=False)(Lign3)
### x-> ( (x^2+x)/2 )^2
Lign5=model_comp(Lign4)

##  x->-x  ###
initializer_2 = tensorflow.keras.initializers.Constant(-1.)
identity_2=layers.Dense(1, activation='linear',
                          kernel_initializer=initializer_2,
                          use_bias=False)(inputs)
### x-> (x^2,-x)
Lign6=layers.concatenate([Lign2, identity_2])
### x-> (-x^2+x)/^2 
Lign7=layers.Dense(1, activation='linear',
                kernel_initializer=init_Z3,
                use_bias=False)(Lign6)
### x-> ( (-x^2+x)/2 )^2
Lign8=model_comp(Lign7)


### fin de la construction
Lign9=layers.concatenate([Lign5, Lign8])

Lign10=layers.Dense(1, activation='linear',
                kernel_initializer=init_Z4,
                use_bias=False)(Lign9)


## affichage
model6=keras.Model(inputs=inputs,outputs=Lign10)
y_predict_new=model6.predict(x_p)   
y_true=x_p**3 
plt.plot(x_p,y_true,'-',label='x^3')
plt.plot(x_p,y_predict_new,'-',label='f(x)')
plt.xlabel('x') 
plt.ylabel('') 
plt.legend(loc="upper left")
plt.show()


## calcul norme erreur
normeLinfty=0
for i in range(0,n):
    normeLinfty=max( normeLinfty, abs(y_predict_new[i]- x_p[i]**3) )
print ("normeLinfty= ",normeLinfty)




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
from keras.callbacks import LearningRateScheduler ##callbacks


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

a=2
def Tak(x):
    b=1*( g1(x)/a+g2(x)/a**2+g3(x)/a**3
       +g4(x)/a**4+g5(x)/a**5+g6(x)/a**6 
       +g7(x)/a**7+g8(x)/a**8+g9(x)/a**9 ) #+20*x**5*(1-x)
    #b=x*x
    #b=x-x*x*x*x*x*x*x
    return b

def Tak_2(x):
    b=1*( g1(x)/a+g2(x)/a**2#+g3(x)/a**3
#       +g4(x)/a**4+g5(x)/a**5+g6(x)/a**6 
#       +g7(x)/a**7+g8(x)/a**8+g9(x)/a**9
       ) #+20*x**5*(1-x)
    #b=x*x
    #b=x-x*x*x*x*x*x*x
    return b

def Tak_4(x):
    b=1*( g1(x)/a+g2(x)/a**2+g3(x)/a**3
       +g4(x)/a**4#+g5(x)/a**5+g6(x)/a**6 
#       +g7(x)/a**7+g8(x)/a**8+g9(x)/a**9
       ) #+20*x**5*(1-x)
    #b=x*x
    #b=x-x*x*x*x*x*x*x
    return b

def Tak_6(x):
    b=1*( g1(x)/a+g2(x)/a**2+g3(x)/a**3
       +g4(x)/a**4+g5(x)/a**5+g6(x)/a**6 
#       +g7(x)/a**7+g8(x)/a**8+g9(x)/a**9
       ) #+20*x**5*(1-x)
    #b=x*x
    #b=x-x*x*x*x*x*x*x
    return b

n=4000
x_petit=np.linspace(0,1,n)
y_petit=Tak_2(x_petit)
w_petit=Tak_4(x_petit)
t_petit=Tak_6(x_petit)
z_petit=Tak(x_petit)

plt.figure(dpi=600)


plt.plot(x_petit,y_petit,label='Tak2')
plt.plot(x_petit,w_petit,label='Tak4')
plt.plot(x_petit,t_petit,label='Tak6')

#plt.plot(x_petit,y_petit,'-',label='Tak')
plt.legend(loc="upper right")
plt.show()


##callbacks
## Definition des callbacks et du learning_rate_scheduler
##callbacks 

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
    #return 0.001/((np.abs(epoch-200)+1)**0.5)
    return 0.001/(max(epoch-200,1)**0.5)
    #return 0.01+epoch/3000*(0.00001-0.01) 

history = LossHistory()
historyepoch = LossEpochHistory()

historyval = VallLossHistory()
historylearnrate=LearnrateEpochHistory()
lrate = LearningRateScheduler(scheduler,verbose=1)

##callbacks
## Fin definition  callbacks/learning_rate_scheduler
##callbacks 

n=100
x_p=np.linspace(0,1,n)
y_p=g2(x_p)
z_p=Tak(x_p)

plt.figure(dpi=600)


#plt.plot(x_p,z_p,label='fobj(x)')

print("Fin partie 1")
#sys.exit(0)




print("\n------------------------------------------\n")

start = dt.datetime.now()

SEED = 2021; np.random.seed(SEED); tensorflow.compat.v1.random.set_random_seed(SEED)

# Loading datasets
DATA_DIR = "../DATA/"
train_data = np.loadtxt(DATA_DIR + "TRAIN_tak.txt", dtype=np.float32, delimiter=',')
np.random.shuffle(train_data)
x_train = train_data[:, 0:1]
y_train = train_data[:, 1:2]

print (train_data)
#print (y_train)


print ("Taille x_train=",x_train.shape)
print ("Taille y_train=",y_train.shape)

test_data = np.loadtxt(DATA_DIR + "TEST_tak.txt", dtype=np.float32, delimiter=',')
np.random.shuffle(test_data)
x_test = test_data[:, 0:1]
y_test = test_data[:, 1:2]


print ("Taille x_test=",x_test.shape)
print ("Taille y_test=",y_test.shape)



#--- Fonction activatin T-ReLU -------#


def T_relu(x):
    return K.relu(x, max_value=1)

#inializer=  tensorflow.keras.initializers.RandomUniform(minval=0., maxval=10.)

initio= tensorflow.keras.initializers.RandomUniform(
     minval=-0.05, maxval=0.05, seed=None)

def L_relu(x):
    return K.relu(x, alpha=0.5, max_value=None, threshold=1)

leaky_relu_alpha = 0.5

lrelu = lambda x: keras.activations.relu(x, alpha=0.5)

#------ Defining model -------#
model = Sequential()
model.add(Dense(3, input_dim=1,name="couche_entree",
                kernel_initializer='random_uniform',
                use_bias=True, bias_initializer='random_uniform',
                activation=lrelu))

model.add(Dense(3,name="couche_hidden_1",
                kernel_initializer='random_uniform',
                use_bias=True, bias_initializer='random_uniform',
                activation=lrelu))



model.add(Dense(3,name="couche_hidden_2",
                kernel_initializer='random_uniform',
                use_bias=True, bias_initializer='random_uniform', 
                activation=lrelu))

model.add(Dense(3,name="couche_hidden_5",
                kernel_initializer='random_uniform',
                use_bias=True, bias_initializer='random_uniform', 
                activation=lrelu))

model.add(Dense(3,name="couche_hidden_6",
                kernel_initializer='random_uniform',
                use_bias=True, bias_initializer='random_uniform', 
                activation=lrelu))

# model.add(Dense(3,name="couche_hidden_7",
#                 kernel_initializer='random_uniform',
#                 use_bias=True, bias_initializer='random_uniform', 
#                 activation=lrelu))


# model.add(Dense(3,name="couche_hidden_8",
#                 kernel_initializer='random_uniform',
#                 use_bias=True, bias_initializer='random_uniform', 
#                 activation=lrelu))

# model.add(Dense(3,name="couche_hidden_5",
#                 kernel_initializer=init_W5,
#                 use_bias=True, bias_initializer=init_b5, 
#                 activation='relu'))

model.add(Dense(1,name="couche_sortie",
                kernel_initializer='random_uniform',
                use_bias=True, bias_initializer='random_uniform', 
                activation='linear'))

Wini=model.get_weights()


model.compile(loss='mse', optimizer=Adam())
#model.compile(loss='mse', optimizer=SGD(learning_rate=0.01, 
#                                         momentum=0.1, 
#                                         nesterov=True))

model.summary()

y_predict_ini=model.predict(x_p)    
#plt.plot(x_p,y_predict_ini,'-',label='f3(x)')
#plt.xlabel('x') 
#plt.ylabel('') 
#plt.legend(loc="upper left")


print("Fin initialisation 2")
#sys.exit(0)



# Learning
toto=model.fit(x_train, y_train,
		  batch_size=128,
		  epochs=000,
          callbacks=[history,historyepoch,historyval,
                     lrate,historylearnrate],##callbacks
		  verbose=1, validation_data=(x_test, y_test)
		  ) 

Wfin=model.get_weights()


rmse = np.sqrt(model.evaluate(x_train, y_train, verbose=False))
print('Test l2-error: {:.6f}'.format(rmse))
y_predict=model.predict(x_p)    
diff_y=np.linspace(0,1,n)
for i in range(0,n):
        diff_y[i]=z_p[i]-y_predict[i]


#plt.plot(x_p,y_predict,'-',label='f(x)')
plt.show()

#------------  graphe fonction-----------#
string_file='%5.8f %5.8f %5.8f %5.8f'
string_file=string_file+' \n'

strstr ='./sortie.plot'
fichier = open(strstr,'w')
for i in range(0,n):
    fichier.write(string_file %  
                  (x_p[i],z_p[i],y_predict_ini[i],y_predict[i]) )
fichier.close()

##callback
##callback
##callback
#------------  sortie historyepoch -----------#
string_file='%5.8f'
string_file=string_file+' \n'
strstr ='./historyepoch.txt'
fichier = open(strstr,'w')
for i in range(0,len(historyepoch.losses)):
        fichier.write(string_file %  (historyepoch.losses[i]))
        #print (i,history.losses[i])
fichier.close() 
#------------  sortie historyall (suivant les batches) -----------#
strstr ='./historyall.txt'
fichier = open(strstr,'w')
for i in range(0,len(history.losses)):
        fichier.write(string_file %  (history.losses[i]))
        #print (i,history.losses[i])
fichier.close() 

#------------  sortie learingrate -----------#
strstr ='./historylearnrate.txt'
fichier = open(strstr,'w')
for i in range(0,len(historylearnrate.losses)):
        fichier.write(string_file %  (historylearnrate.losses[i]))
fichier.close() 

end = dt.datetime.now()
print("Training duration: {} seconds".format(end - start))


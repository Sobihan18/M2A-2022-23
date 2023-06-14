import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import datetime as dt

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import time 


print("\n------------------------------------------\n")

start = dt.datetime.now()

SEED = 2020; np.random.seed(SEED); tf.compat.v1.random.set_random_seed(SEED)

# Loading datasets
DATA_DIR = "../DATA/"
train_data = np.loadtxt(DATA_DIR + "TRAIN.txt", dtype=np.float32, delimiter=',')
np.random.shuffle(train_data)
x_train = train_data[:, 2:11]
y_train = train_data[:, 0:2]

test_data = np.loadtxt(DATA_DIR + "TEST.txt", dtype=np.float32, delimiter=',')
np.random.shuffle(test_data)
x_test = test_data[:, 2:11]
y_test = test_data[:, 0:2]

#print ("x_test")
#print (x_test)

# Defining model
model = Sequential()
model.add(Dense(40, use_bias=True, bias_initializer='random_uniform', activation='relu', input_dim=x_train.shape[1]))
model.add(Dense(20, use_bias=True, bias_initializer='random_uniform', activation='relu'))
model.add(Dense(2,  use_bias=True, bias_initializer='random_uniform', activation='linear'))
model.compile(loss='mse', optimizer=Adam())



# Learning
model.fit(x_train, y_train,
		  batch_size=1024,
		  epochs=100,
		  verbose=1, validation_data=(x_test, y_test)
		  ) 

rmse = np.sqrt(model.evaluate(x_test, y_test, verbose=False))
print('Test l2-error: {:.6f}'.format(rmse))


print (' ')
y_essai=model.predict(x_test,verbose=1)    
for i5 in range(0,2): 
    print ('1  : y_predict',y_essai[[i5]],': y_vrai   ',y_test[[i5]]) #,': x  ',x_test[[i5]])
    
    
data = np.array([[1,1,1,0.5,0.5,0.5,0,0,0]])
prediction = model.predict(data)
print("prediction=",prediction)    
    

end = dt.datetime.now()
print("Training duration: {} seconds".format(end - start))

#from keras2cpp import export_model
#export_model(model, 'interface.model')


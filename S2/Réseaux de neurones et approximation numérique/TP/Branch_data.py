"""
Spyder Editor

This is a temporary script file.
"""


import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from random import shuffle
from random import seed
from random import *
import shutil
import math
import pickle
import string
import numpy
#import numba
#from numba import jit
import time 
import sys
import numpy as np
#import matplotlib.pyplot as plt


seed()
Delta=0.

##########################
###### Parameters ########
##########################

print("-------------")
print("-------------")
print("-------------")
print("-------------")



def spec(x):
    #4*x**3-3*x #x-x*x #
    a=(x+1)/2.
    return a-a*a #16*x**5 -20*x**3+ 5*x #4*x**3-3*x #a**3 #
    #8*(x**4)-8*(x**2)+1#16*x**5 -20*x**3+ 5*x#8*(x**4)-8*(x**2)+1 


M=10000

string_file='%5.8f,%5.8f,%5.8f'+' \n'
strstr_train='./TRAIN_tak.txt'
fichier_train = open(strstr_train,'w')
strstr_test='./TEST_tak.txt'
fichier_test  = open(strstr_test,'w')
strstr_train='./IMP_tak.txt'
fichier_imp = open(strstr_train,'w')



for i in range(0,M):
    a=i*1./(M-1)
    x=2*a-1
    b=spec(x)
           
    if (np.random.uniform(0., 1.)>0.2):
        fichier_train.write(string_file %  (a,b,a) )
    else:
        fichier_test.write(string_file %  (a,b,a) )

for i in range(0,101):
    a=i*1./(100)
    x=2*a-1
    b=spec(x)
    fichier_imp.write(string_file %  (a,b,a) )

 
fichier_train.close()
fichier_test.close()
fichier_imp.close()
print("This is the end")
    

        
        
        
        
        

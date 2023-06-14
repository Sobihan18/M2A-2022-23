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
import matplotlib.pyplot as plt


seed()
Delta=0.

##########################
###### Parameters ########
##########################

print("-------------")
print("-------------")
print ("Takagi function")
print("-------------")
print("-------------")

def g1(x): return 2*x*(x<0.5) + (2-2*x)*(x>=0.5)
def g2(x): return g1(g1(x))
def g3(x): return g1(g2(x))
def g4(x): return g1(g3(x))
def g5(x): return g1(g4(x))
def g6(x): return g1(g5(x))
def g7(x): return g1(g6(x))
def g8(x): return g1(g7(x))
def g9(x): return g1(g8(x))

at=4
def Tak(x):
    b=1*( g1(x)/at+g2(x)/at**2+g3(x)/at**3
       +g4(x)/at**4+g5(x)/at**5+g6(x)/at**6 
       +g7(x)/at**7+g8(x)/at**8+g9(x)/at**9 ) #+ 20*x**5*(1-x)
    b=x-x*x*x*x*x*x*x
    return b

def spec(x):
    return np.exp(-(x-0.5)**4)


N=1 # Ncell=2N+1: N=0,1, ou 2
M=4000

x_p=np.linspace(0,1,M); y_p=Tak(x_p); 
plt.figure(dpi=600)
plt.plot(x_p,y_p)
plt.xlabel('x') 
plt.ylabel('f(x)') 
plt.show()


string_file='%5.8f,%5.8f,%5.8f'
string_file=string_file+' \n'

strstr_train='../DATA/TRAIN_tak.txt'
fichier_train = open(strstr_train,'w')
strstr_test='../DATA/TEST_tak.txt'
fichier_test  = open(strstr_test,'w')



for i in range(0,M):
    a=np.random.uniform(0,1)
    a=(1+np.cos( i/(M-1.) *3.1415926) )/2.
    b= (a*a)*1. 
    c=(1-2*a)*4.
    b=Tak(a)
    b=a-a*a*a*a*a*a*a
    #b=spec(a)
    
#    eps=0.0001                
    if (np.random.uniform(0., 1.)>0.2):
        fichier_train.write(string_file %  (a,b,a) )
        #print (a,b,a)
    else:
        fichier_test.write(string_file %  (a,b,a) )

 
fichier_train.close()
fichier_test.close()
print("This is the end")
    

        
        
        
        
        

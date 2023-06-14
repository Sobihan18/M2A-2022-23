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
    #b=x-x*x*x*x*x*x*x
    return b

def spec(x):
    #4*x**3-3*x #x-x*x #
    a=(x+1)/2.
    return 1-np.cos(1*6.28*x)
    #16*x**5 -20*x**3+ 5*x #4*x**3-3*x #a**3 #a-a*a
    #8*(x**4)-8*(x**2)+1#16*x**5 -20*x**3+ 5*x#8*(x**4)-8*(x**2)+1 


N=1 # Ncell=2N+1: N=0,1, ou 2
M=10000

x_p=np.linspace(0,1,M); y_p=Tak(x_p); 
# plt.figure(dpi=600)
# plt.plot(x_p,y_p)
# plt.xlabel('x') 
# plt.ylabel('f(x)') 
# plt.show()


#string_file='%5.8f %5.8f %5.8f\n'
string_file='%5.8f,%5.8f,%5.8f'+' \n'

strstr_train='../DATA/TRAIN_tak.txt'
fichier_train = open(strstr_train,'w')
strstr_test='../DATA/TEST_tak.txt'
fichier_test  = open(strstr_test,'w')
strstr_train='../DATA/IMP_tak.txt'
fichier_imp = open(strstr_train,'w')



for i in range(0,M):
    a=i*1./(M-1)
    x=2*a-1
    b=Tak(a)
           
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
    

        
        
        
        
        

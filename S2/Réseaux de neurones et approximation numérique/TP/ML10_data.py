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

seed()
Delta=0.

##########################
###### Parameters ########
##########################

print("-------------")
print("-------------")
print("-------------")
print("-------------")


def f1(x):
    return np.sin(x)
def f2(a,x):
    return a*x
def f3(a,b,x):
    return a*x+b*x*x

def f4(a,b,c,x):
    return a*x+b*x*x+c*x*x*x


N_interp=3 # Ncell
N_debut=-1
N_fin=N_debut+N_interp-1
x_points= np.linspace(N_debut,N_fin,N_interp)
x_inputs= np.zeros(N_interp)
y_output=0.

M=150000
Mdx=100



string_file_uni='%5.8f'
string_file_virg=','
string_file_term=' \n'

string_file='%5.8f,%5.8f,%5.8f,%5.8f'
string_file=string_file+' \n'

strstr_train='../DATA/TRAIN_tak.txt'
fichier_train = open(strstr_train,'w')
strstr_test='../DATA/TEST_tak.txt'
fichier_test  = open(strstr_test,'w')



for i in range(0,M):
    dx=np.random.uniform(0,1.2)
    a=np.random.uniform(-0.3,0.3)
    b=np.random.uniform(-0.3,0.3)

    x_inputs=f2(a,x_points)
    y_output=f2(a,-dx)
    
    x_inputs=f3(a,b,x_points)
    y_output=f3(a,b,-dx)
    
 #   print("a=",a,", x=",x_inputs,"||   -dx=",-dx,", y=",y_output)
    
    if (np.random.uniform(0., 1.)>0.2):
        fichier_train.write(string_file_uni %  (y_output) )
        fichier_train.write(string_file_virg)
        fichier_train.write(string_file_uni %  (-dx) )

        for k in range(0,N_interp):
            fichier_train.write(string_file_virg)
            fichier_train.write(string_file_uni %  (x_inputs[k]) )
        fichier_train.write(string_file_term)
    else:
        fichier_test.write(string_file_uni %  (y_output) )
        fichier_test.write(string_file_virg)
        fichier_test.write(string_file_uni %  (-dx) )
        for k in range(0,N_interp):
            fichier_test.write(string_file_virg)
            fichier_test.write(string_file_uni %  (x_inputs[k]) )
        fichier_test.write(string_file_term)
 
fichier_train.close()
fichier_test.close()
print("This is the end")
    

        
        
        
        
        

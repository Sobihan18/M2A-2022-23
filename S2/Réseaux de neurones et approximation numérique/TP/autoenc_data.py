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


N_interp=3 # Ncell
N_debut=-1
N_fin=N_debut+N_interp-1
x_points= np.linspace(N_debut,N_fin,N_interp)
x_inputs= np.zeros(N_interp)
y_output=0.

M=5000
taille_x=150
taille_theta=7

Mat= 10*np.random.rand(taille_x,taille_theta)

X_vec=np.ones(taille_x)



# print ("Mat=",Mat)
# print ("Theta_vec=",Theta_vec)
# print ("X_vec=",X_vec)





string_file_uni='%5.8f'
string_file_virg=','
string_file_term='\n'
strstr_auto='./DATA/TRAIN_auto.txt'
fichier_auto = open(strstr_auto,'w')

for i in range(0,M):

    Theta_vec=np.random.rand(taille_theta)
    X_vec= Mat@Theta_vec

    fichier_auto.write(string_file_uni %  (X_vec[0]) )
    fichier_auto.write(string_file_virg)

    for k in range(1,taille_x-1):
        fichier_auto.write(string_file_uni %  (X_vec[k]) )
        fichier_auto.write(string_file_virg)
        
    fichier_auto.write(string_file_uni %  (X_vec[taille_x-1]) )
    fichier_auto.write(string_file_term)
    
fichier_auto.close()

print("This is the end")
    

        
        
        
        
        

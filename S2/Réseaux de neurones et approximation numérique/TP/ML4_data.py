"""
Produce the data for the classification of curves
B. Despres
01/11/2021
"""


import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from random import seed
from random import *
import random 
import shutil
import math
import pickle
import string
import numpy
#import numba
#from numba import jit

import numpy as np
import matplotlib.pyplot as plt


seed()
Delta=0.

##########################
###### Parameters ########
##########################

print("-------------")
print("-------------")
print ("Fonctions oscillantes")
print("-------------")
print("-------------")



N=10000

nombre_omega=10#10
omegas=np.zeros(nombre_omega)
phases=np.zeros(nombre_omega)
amplitudes=np.zeros(nombre_omega)
labels=np.zeros(nombre_omega)

M=100
x=np.linspace(0,1,M)
y=np.zeros(M)


pi=3.1415926

#plt.figure(dpi=600)



string_file='%5.4f,'
string_file_int='%2.d,'
string_line_end=' \n'

strstr_train='../DATA/FORMAT_curves.txt'
fichier_format = open(strstr_train,'w')
fichier_format.write('%2.d,' %nombre_omega)
fichier_format.write('%2.d' %M)
fichier_format.close()


strstr_train='../DATA/TRAIN_curves.txt'
fichier_train = open(strstr_train,'w')
strstr_test='../DATA/TEST_curves.txt'
fichier_test  = open(strstr_test,'w')


for i in range(0,N):
    labels=np.zeros(nombre_omega)
    n=random.randint(1,nombre_omega)
    labels[n-1]=1
    #labels[0]=1

    
    for j in range(0,n):
        omegas[j]=j+1#random.uniform(0,1)
        phases[j]=uniform(0,1)
        amplitudes[j]=random.uniform(0.2,1)
        
        # omegas[j]=random.uniform(0,1)
        # phases[j]=uniform(0,1)
        # amplitudes[j]=random.uniform(0.8,1)
        
    for k in range(0,M):
        y[k]=0.
        
        for j in range(0,n):
            y[k]=y[k]+np.cos(2*pi*( omegas[j]*x[k]+phases[j] ))*amplitudes[j]
        
    plt.plot(x,y)
    #print (y)
    #print (N,M,n)

            
#------ Ecriture fichiers ------#            

    
    if (np.random.uniform(0., 1.)>0.2):
        for j in range(0,nombre_omega):
            fichier_train.write(string_file % labels[j])
        for k in range(0,M):
            fichier_train.write(string_file % y[k]) 
        fichier_train.write('%5.4f' % -100) 
        fichier_train.write(string_line_end)        

    else:
        for j in range(0,nombre_omega):
            fichier_test.write(string_file % labels[j])
        for k in range(0,M):
            fichier_test.write(string_file % y[k]) 
        fichier_test.write('%5.4f' % -100) 
        fichier_test.write(string_line_end)        
 
fichier_train.close()
fichier_test.close()
print("This is the end")
    

        
        
        
        
        

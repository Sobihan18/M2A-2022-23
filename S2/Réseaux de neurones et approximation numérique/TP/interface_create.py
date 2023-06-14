
# coding: utf-8

# In[6]:

#!/Library/Frameworks/Python.framework/Versions/2.7/bin/python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from random import shuffle
from random import seed
from random import *
import shutil
import math
import pickle
import string

import numpy

import time 


import numpy as np

seed()
Delta=0.



print ("Interfaces droites")

##########################
###### Parameters ########
##########################

N=1 # Ncell=2N+1: N=0,1, ou 2
print ("Images= ",2*N+1,"x",2*N+1)


Nx=100 # Nomre points de quadrature par maille: recommandation: pair
a=0

M=360*2 # dcretisation angle
P=3*3  # discretisation rayon sur 2P+1 points
R=(N+0.5)*2**0.5
pi2=2*3.1415926




#@jit(nopython=True)
def integration(i1,i2,Nx,c_theta,s_theta,r):
    alpha=0.
    for i5 in range(0,Nx):
        x=(i5+0.5)/Nx-0.5 +i1
        plan_x=x*c_theta-r
        plan_x_moins=-plan_x-(-0.5 +i2)*s_theta
        plan_x_moins=plan_x_moins*Nx- 0.5*s_theta                   
        
        for i6 in range(0,Nx):
            plan_y= i6*s_theta
            alpha+=(plan_y>=plan_x_moins)
        
    return alpha




###################################@
    

temp_array=np.zeros((2*N+1,2*N+1))
temp_array_deca=np.zeros((2*N+1,2*N+1))
temp_array_int=np.zeros((Nx+1))
temp_array_int_deca=np.zeros((Nx+1))



string_file='%5.8f,%5.8f'
for i1 in range(0,(2*N+1)**2):
    string_file=string_file+',%5.4f'
string_file=string_file+' \n'

string_train='../DATA/TRAIN.txt'
fichier_train = open(string_train,'w')


string_test='../DATA/TEST.txt'
fichier_test = open(string_test,'w')


        
Time_exec=0.
Time_deb= time.time()
    

for i3 in range(0,M):
    theta=pi2*i3/M

    c_theta=math.cos(theta)
    s_theta=math.sin(theta)
    
    if (i3%1==0):
        print (i3,': Angle=',theta)
               
    for i4 in range(0,2*P+1):
        distance=np.random.uniform( -(1+0.5)**0.5,(1+0.5)**0.5 )

        for i1 in range(-N,N+1):
            for i2 in range(-N,N+1):                
                    Time_exec_deb= time.time()
                    temp_array[i1+N][i2+N]=integration(i1,i2,Nx,c_theta,s_theta,distance)/(Nx**2)
                    Time_exec_fin= time.time()                   
                    Time_exec=Time_exec+Time_exec_fin-Time_exec_deb
                    
 
        if (N==1):        
                SD=(math.cos(theta),math.sin(theta),temp_array[0][0], temp_array[0][1],temp_array[0][2],
                                                  temp_array[1][0], temp_array[1][1],temp_array[1][2], 
                                                  temp_array[2][0], temp_array[2][1],temp_array[2][2] )
        elif (N==2):
                SD=(math.cos(theta),math.sin(theta),
                                        temp_array[0][0], temp_array[0][1],temp_array[0][2],temp_array[0][3],temp_array[0][4],
                                        temp_array[1][0], temp_array[1][1],temp_array[1][2],temp_array[1][3],temp_array[1][4],
                                        temp_array[2][0], temp_array[2][1],temp_array[2][2],temp_array[2][3],temp_array[2][4],
                                        temp_array[3][0], temp_array[3][1],temp_array[3][2],temp_array[3][3],temp_array[3][4],
                                        temp_array[4][0], temp_array[4][1],temp_array[4][2],temp_array[4][3],temp_array[4][4])
        else:           
                SD=(theta,math.cos(theta),math.sin(theta),temp_array[0][0])
                
        if (np.random.uniform(0., 1.)>0.2):
                    fichier_train.write(string_file %  (SD) )

        else:
                    fichier_test.write(string_file %  (SD) )
                    


fichier_train.close()
fichier_test.close()


               
Time_fin= time.time()
print ('Time_fin=',Time_fin)

             

print ('#(Dataset)= ',M*(2*P+1))


print ('Time total = ',Time_fin-Time_deb)
print ('Time boucle= ',Time_exec)
print ('This is the end')

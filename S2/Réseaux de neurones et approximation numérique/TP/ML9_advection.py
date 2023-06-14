# 1D ADVECTION EQUATION with FD #
# 1D ADVECTION EQUATION with FD #
# 1D ADVECTION EQUATION with FD #

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random

import tensorflow
import tensorflow.keras as keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Input, Dense, Dropout, Reshape,Flatten,BatchNormalization, Activation,ReLU
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras import initializers
from keras import utils
from keras.utils.generic_utils import get_custom_objects


# a= 1, vitesse est normalisée

# model = keras.models.load_model('DATA/my_model')

# x_mpl=np.array([np.array(4*[np.float32(0.)])])

# x_mpl[0][0]=-0.3
# x_mpl[0][1]=1
# x_mpl[0][2]=0
# x_mpl[0][3]=-1
# y_sortie=model.predict(x_mpl,verbose=1) 
# print("x=",x_mpl,", y=",y_sortie)
   
#------ Parametres physiques ---------#
Coeff = 1.  #coefficient physique
Lx = 1.0  #taille du domaine
T = 1 #temps d'integration
print("T final=",T)


#-------- Parametres numeriques-----#
NX = 101  #nombre de points de grille
CFL=0.71234567 #0.7187654   # condition CFL de stabilité  0<CFL<= 1
dx = Lx/(NX-1) #pas d'espace
dt = CFL*dx    #pas de temps
NT = int(T/dt)  #nombre de pas de temps
pi=3.14159265358979323846
random.seed()

print("Nombre pas de temps= ",NT)
print("dt= ",dt)


#------- DEFINION GRILLE ESPACE --#
xx = np.zeros(NX)
for i in np.arange(0,NX):
      xx[i]=i*dx


#------- Declaration des variables physiques  --#
ddU   = np.zeros(NX)
U_data = np.zeros(NX)
U_old = np.zeros(NX)
U_int = np.zeros(NX)
U_new = np.zeros(NX)

#------- Initialisation donnée initiale  --#
U_data = np.cos(2*pi*xx)    # fonction cosinus
for j in np.arange(0,NX): # Creneau
    U_data[j]=0.         
    if (0.25<xx[j]<0.75): 
        U_data[j]=1.

for i in np.arange(0,NX):
    U_old[i]= U_data[i]
    U_int[i]= U_data[i]


plt.figure()
plt.plot(xx,U_old,label="t=0.")
plt.xlabel('x')
plt.ylabel('u')
plt.legend(loc="upper left")

plt.show()



#--------------------#
# Boucle en temps ---#
#--------------------#
time=0.
for n in np.arange(0,NT):
    
    time=time+dt    
    if (n%10==0):
        print ("t=",time)
 #       print ("U_old, begin0",U_old)
        

# Schema Upwind      (le défault)  
    for j in np.arange(1,NX-1):
        ddU[j] = U_old[j]-U_old[j-1]       
    ddU[0] = U_old[0]-U_old[NX-2]   # cas au bord périodique
    

# #Schema Centered instable
#    for j in np.arange(1,NX-1):
#        ddU[j] = (U_old[j+1]-U_old[j-1])/2.
#    ddU[0] = (U_old[1]-U_old[NX-2])/2.
    
#Schema Lax-Wendroff stable
    for j in np.arange(1,NX-1):
        ddU[j] = (U_old[j+1]-U_old[j-1]+CFL*(2*U_old[j]-U_old[j+1]-U_old[j-1]))/2.
    ddU[0] = (U_old[1]-U_old[NX-2]+CFL*(2*U_old[0]-U_old[1]-U_old[NX-2]))/2.    
    
         
#---------- Actualisation   U_new, U_old
    for j in np.arange(0,NX-1):
        U_new[j]=U_old[j]-(dt/dx)*ddU[j]    
    U_new[NX-1]=U_new[0]   
    U_old=U_new
    
    plt.figure()
    plt.plot(xx,U_new,"r",marker='x',label="t=T")
    plt.pause(0.1) # pause avec duree en secondes
    plt.show()
    
#--------------------#
# Fin boucle en temps ---#
#--------------------#
print ("tFinal=",time)


#-----  Affichage resultat ----#

plt.figure()
plt.plot(xx,U_int,label="t=0.")
plt.plot(xx,U_new,"r",marker='x',label="t=T")
plt.legend(loc="upper left")
plt.xlabel('x')
plt.ylabel('u')
plt.show()





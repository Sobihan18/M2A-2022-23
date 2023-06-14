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

model = keras.models.load_model('../DATA/my_model')

x_mpl=np.array([np.array(4*[np.float32(0.)])])

x_mpl[0][0]=-0.5
x_mpl[0][1]=1
x_mpl[0][2]=0
x_mpl[0][3]=-1


y_sortie=model.predict(x_mpl,verbose=1) 
print("x=",x_mpl,", y=",y_sortie)
   
#------ Parametres physiques ---------#
Coeff = 1.  #coefficient physique
Lx = 1.0  #taille du domaine
T = 1. #temps d'integration
print("T final=",T)


#-------- Parametres numeriques-----#
NX = 51  #nombre de points de grille
CFL=.6 #0.7187654   # condition CFL de stabilité  0<CFL<= 1
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

xx_res = np.zeros(NX)
for i in np.arange(0,NX):
      xx_res[i]=i*dx-T


#------- Declaration des variables physiques  --#
ddU   = np.zeros(NX)
U_data = np.zeros(NX)
U_old = np.zeros(NX)
U_int = np.zeros(NX)
U_new = np.zeros(NX)

U_ml_input=np.array(NX*[np.array(4*[np.float32(0.)])])
U_ml_output=np.array(NX*[np.array(1*[np.float32(0.)])])
print ("Taille U_ml_input=",U_ml_input.shape)
for j in np.arange(0,NX): 
    U_ml_input[j][0]=-(1+CFL)/2. #-CFL
    #U_ml_input[j][0]=-CFL


#------- Initialisation donnée initiale  --#
U_data = np.cos(2*pi*xx)    # fonction cosinus
U_res = np.cos(2*pi*xx_res)    # fonction cosinus
# for j in np.arange(0,NX): # Creneau
#     U_data[j]=0.         
#     if (0.25<xx[j]<0.75): 
#         U_data[j]=1.

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
        masse=0.
        for j in np.arange(0,NX):
            masse=masse+U_old[j]*dx
        print ("t=",time)

            
 #       print ("U_old, begin0",U_old)
        

# Schema Upwind      (le défault)  
    # for j in np.arange(1,NX-1):
    #     ddU[j] = U_old[j]-U_old[j-1]       
    # ddU[0] = U_old[0]-U_old[NX-2]   # cas au bord périodique
    

# #Schema Centered instable
#    for j in np.arange(1,NX-1):
#        ddU[j] = (U_old[j+1]-U_old[j-1])/2.
#    ddU[0] = (U_old[1]-U_old[NX-2])/2.
    
#Schema Lax-Wendroff stable
    # for j in np.arange(1,NX-1):
    #     ddU[j] = (U_old[j+1]-U_old[j-1]+CFL*(2*U_old[j]-U_old[j+1]-U_old[j-1]))/2.
    # ddU[0] = (U_old[1]-U_old[NX-2]+CFL*(2*U_old[0]-U_old[1]-U_old[NX-2]))/2.    
    
#Schema Machine Learning        
    for j in np.arange(1,NX): 
        U_ml_input[j][1]=U_old[j-1]-U_old[j]
    U_ml_input[0][1]=U_old[NX-2]-U_old[0]    
    
    for j in np.arange(0,NX): 
        U_ml_input[j][2]=0 #U_old[j]
    
    for j in np.arange(0,NX-1): 
        U_ml_input[j][3]=U_old[j+1]-U_old[j]
    U_ml_input[NX-1][3]=U_old[1]-U_old[NX-1]
    
    U_ml_output=model.predict(U_ml_input,verbose=0) 
    
    for j in np.arange(0,NX):
        U_ml_output[j]=U_ml_output[j]+U_old[j]
    
    for j in np.arange(0,NX-1):
        ddU[j] = U_ml_output[j+1]-U_ml_output[j]       
    ddU[NX-1] = ddU[0]  # cas au bord périodique
    
    
    
    
         
#---------- Actualisation   U_new, U_old

    masse_old=0.
    masse_new=0.
    for j in np.arange(0,NX-1):
        #U_new[j]=U_ml_output[j]
        U_new[j]=U_old[j]-(dt/dx)*ddU[j]    
        #U_new[j]=U_ml_output[j]
        masse_old=masse_old+U_old[j]*dx
        masse_new=masse_new+U_new[j]*dx
    #print ("masse_old=",masse_old,"masse_new=",masse_new)
        
    for j in np.arange(1,NX-1):
        m_j=min(U_old[j],U_old[j-1])
        M_j=max(U_old[j],U_old[j-1])
        #U_new[j]=min( max(m_j,U_new[j]),M_j )
        
    m_j=min(U_old[0],U_old[NX-2])
    M_j=max(U_old[0],U_old[NX-2])    
    #U_new[0]=min( max(m_j,U_new[0]),M_j )
    
    for j in np.arange(1,NX-1):
        if (U_new[j]>1.01):
            m_j=min(U_old[j],U_old[j-1])
            M_j=max(U_old[j],U_old[j-1])
           # print ("PG, n: j",n,j,m_j,M_j,U_old[j],U_old[j-1])
            
  #  print (n,":U_old",U_old)
   # print (n,":U_new",U_new)
            
        
    U_new[NX-1]=U_new[0]  
    for j in np.arange(0,NX):
        U_old[j]=U_new[j]
    
    if (n%10==0):
            plt.figure()
            plt.plot(xx,U_new,"r",label="t=T")
            plt.pause(0.1) # pause avec duree en secondes
            plt.show()
    
#--------------------#
# Fin boucle en temps ---#
#--------------------#
print ("tFinal=",time)

norml2=0.
for j in np.arange(0,NX):
    norml2=norml2+(U_new[j]-U_res[j])**2
norml2=np.sqrt( norml2*dx )
print("Erreur L2=",norml2)


#-----  Affichage resultat ----#



plt.figure()
plt.figure(dpi=1200)

plt.plot(xx,U_res,"b",label="exact sol.")
plt.plot(xx,U_new,"r","+",label="numerical sol.")
plt.legend(loc="upper right")
plt.xlabel('x')
plt.ylabel('u')
plt.savefig('foo.png')
plt.show()








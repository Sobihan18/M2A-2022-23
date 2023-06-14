#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 13:42:50 2022

@author: despres
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 16:50:27 2021

@author: despres
"""


import numpy as np
import random
import matplotlib.pyplot as plt
import sys
import math
#import seaborn as sns
#import pandas as pd

random.seed(a=None, version=2)


def f(dim,x):
    a=0
    for l in range(dim):
       a=a+x[l]
    t=np.exp(a)
    return t

  
dim=15
N=100
MC=10000
hisy_lh=np.arange(0,MC,dtype=float)
hisy_mc=np.arange(0,MC,dtype=float)

     
strstr_latin='./out_latin.txt'
fichier_latin = open(strstr_latin,'w')
string_file='%5.4f %5.4f %5.4f %5.0f %5.0f %5.0f %5.0f \n'


for mc in range (MC):
    print ("mc=",mc)

           
    Sigma    =np.random.randint(N, size=(dim, N))
    Sigma_inv=np.random.randint(N, size=(dim, N))


    amat=np.arange(0,N)
    for i in range(dim):
        random.shuffle(amat)
        for j in range(N):
            Sigma[i][j]=amat[j]
 

    Integral_rank1=0.
    Integral_MC=0.



    a=np.zeros(dim,dtype=int)
    b=np.zeros(dim,dtype=float)


    for i in range(N):
            # Hypercube Latin
            # boucle importante
            for l in range(dim):
                    a[l]=i
                    b[l]=Sigma[l][a[l]]*1./N+1./(2*N) 
            res=f(dim,b)
            Integral_rank1=Integral_rank1+res
            #print (Integral_rank1)

            # Monte-Carlo
            # boucle importante
            b=np.random.rand(dim)        
            res=f(dim,b)
            Integral_MC=Integral_MC+res
        
 
        


    Integral_rank1=Integral_rank1/N
    Integral_MC=Integral_MC/N
            
    hisy_lh[mc]=Integral_rank1
    hisy_mc[mc]=Integral_MC
 
            


print("Integral(2,1)=",Integral_rank1,"ref=",pow(np.exp(1)-1,dim))




fichier_latin.close()

plt.hist(hisy_lh,bins=30)
plt.show()
plt.hist(hisy_mc,bins=30)
plt.show()



print ("THE END")

import os
import datetime as dt
import numpy as np
import tensorflow
from tensorflow.python.client import device_lib
from tensorflow import device
print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))
tensorflow.config.list_physical_devices('GPU') 
import sys
import tensorflow.keras as keras

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from keras import backend as K
tensorflow.autograph.set_verbosity(0)

from tqdm.keras import TqdmCallback
from keras import Input, layers

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam,SGD
from keras.layers import Input, Dense, Dropout, Reshape,Flatten,BatchNormalization, Activation,ReLU
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
# from keras import metrics
from keras import initializers
from tensorflow.keras import initializers,activations
from keras.layers import LeakyReLU
from keras.layers import Lambda

from keras import utils
from keras.utils.generic_utils import get_custom_objects
from keras.callbacks import LearningRateScheduler
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import ast


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

physical_devices = tf.config.list_physical_devices()
for dev in physical_devices:
  print("  DEV=  ",dev)
  
#with tensorflow.device('/GPU:0'):
if (1==0): # Debut 
    
    """
    ## Prepare the data
    """
    num_classes = 10
    input_shape = (28, 28, 1)
        
    def load_data(path):
        with np.load(path) as f:
            x_train, y_train = f['x_train'], f['y_train']
            x_test, y_test = f['x_test'], f['y_test']
            return (x_train, y_train), (x_test, y_test)

    (x_train, y_train), (x_test, y_test) = load_data('./mnist.npz')  
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print (" ")
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    print ("Taille x_train=",x_train.shape)
    print ("Taille y_train=",y_train.shape)    
    
    
    """
    ## Build the model
    """
    def T_relu(x):
        return K.relu(x, max_value=1)
    
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation=T_relu),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation=T_relu),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    
    """
    ## Train the model
    """
    batch_size = 128
    epochs = 10
    #'mse'
    #model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    
    """
    ## Evaluate the trained model
    """
    score = model.evaluate(x_test, y_test, verbose=0)
    
    
    print("  On essaye",x_train.shape)

    y_sortie=model.predict(x_train,verbose=1)
    
    print("  On essaye",x_train.shape,y_sortie.shape)

    
    print(y_sortie[0])
    print(y_test[0])
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])  

with tensorflow.device('/GPU:0'):
    
    print("\n------------------------------------------\n")
    

    
    class LossHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.losses = []
        def on_batch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss'))
            
    class VallLossHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.losses = []
        def on_batch_end(self, batch, logs={}):
            self.losses.append(logs.get('acc'))
            
    class LossEpochHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.losses = []
        def on_epoch_end(self, epoch, logs={}):
            self.losses.append(logs.get('loss'))
            
    class LearnrateEpochHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.losses = []
        def on_epoch_end(self, epoch, logs={}):
            self.losses.append(logs.get('lr'))
            
    
    def scheduler(epoch, lr):
        return 0.001#/(10*epoch+1)**0.5
        #return 0.01+epoch/3000*(0.00001-0.01) 
        
    def customLoss(yTrue, yPred):
        return K.max(K.abs(yTrue - yPred))    
    
    history = LossHistory()
    historyepoch = LossEpochHistory()
    
    historyval = VallLossHistory()
    historylearnrate=LearnrateEpochHistory()
    lrate = LearningRateScheduler(scheduler,verbose=1)
    
    n=101
    x_p=np.linspace(0,1,n)
    
#    xx_p=x_p.reshape((101,1))
#    xxx_p=tf.convert_to_tensor(xx_p)
    #plt.figure(dpi=600)
    
    
    #plt.plot(x_p,z_p,'+-',label='fobj(x)')
    #plt.legend(loc="upper right")
    #plt.show()
    
    #sys.exit(0)
    
    
    
    
    print("\n------------------------------------------\n")
    
    start = dt.datetime.now()
    
    SEED = 2021; np.random.seed(SEED); tensorflow.compat.v1.random.set_random_seed(SEED)
    
    # Loading datasets
    DATA_DIR = "./"
    
    train_data = np.loadtxt(DATA_DIR + "TRAIN_tak.txt", dtype=np.float32, delimiter=',')
    np.random.shuffle(train_data)
    x_train = train_data[:, 0:1]
    y_train = train_data[:, 1:2]
    print ("Taille x_train=",x_train.shape)
    print ("Taille y_train=",y_train.shape)
    
    test_data = np.loadtxt(DATA_DIR + "TEST_tak.txt", dtype=np.float32, delimiter=',')
    np.random.shuffle(test_data)
    x_test = test_data[:, 0:1]
    y_test = test_data[:, 1:2]  
    print ("Taille x_test=",x_test.shape)
    print ("Taille y_test=",y_test.shape)
    print ("  ")
    
    imp_data = np.loadtxt(DATA_DIR + "IMP_tak.txt", dtype=np.float32, delimiter=',')
    x_imp = imp_data[:, 0:1]
    y_imp = imp_data[:, 1:2]
    
    #------ initialiseur couches ReLU ---#
    #------ initialiseur couches ReLU ---#
    
    print('GPU name: ', tensorflow.config.experimental.list_physical_devices('GPU'))
    from tensorflow.python.client import device_lib 
    print("Nom device",device_lib.list_local_devices())
    
    
    ## nombre intervalles ##
    m=2
    ##vnombre polynomes par intervalle ##
    r=1
    ## nombre couches ##
    couches_tot=6
    couches_tot=couches_tot+1
    
    
    cas=1
    
    if (cas==1):
        print ('cas_test=',cas,': f(x)=x-x**2')
    if (cas==2):
        print ('cas_test=',cas,': f(x)=x*x*x')
        m=2
        r=1
    if (cas==3):
        print ('cas_test=',cas,
               ': f(x)=x(1/3-x)(2/3-x)(1-x)')
    if (cas==4):
        print ('cas_test=',cas,': f(x)=4x**3-3x')
    
    
    def init_W0(shape, dtype=None):
        W=[np.ones(m+1)*m]
        return K.constant(W)
    def init_b0(shape, dtype=None):
        b=-np.linspace(-1,m-1,m+1)
        return K.constant(b)
    
    
    
    
    # def init_W20(shape, dtype=None):
    #     W=np.array([[2]])
    #     return K.constant(W)
    
    
    # def init_b20(shape, dtype=None):
    #     b=np.array([0])
    #     return K.constant(b)
    
    def init_W_old(shape, dtype=None):
        b=m+1 #+ m*r
        W=[np.zeros(b)]
        Z=np.transpose(W)    
    #    print (Z)
        return K.constant(Z)
    
    #    array3=np.zeros(N)
    #    for i in range(0,N): array3[i]=f(i*dx)
    #    return K.constant(np.transpose(np.array([array3])))
    
    #--- Fonction activations et initizlization-------#
    
    
    lrelu = lambda x: keras.activations.relu(x, alpha=0.5)
    trelu = lambda x: keras.activations.relu(x, max_value=1)
    inputs = keras.Input(shape=(1),name="inputs")
    
    initializer_0 = tf.keras.initializers.Constant(0.)
    Lign_0=layers.Dense(1, activation=trelu,
                          kernel_initializer=initializer_0,
                          use_bias=False, 
                          trainable=False)(inputs)
    model_old = keras.Model(inputs=inputs, 
                        outputs=Lign_0)
    
    #model_old.summary()
    
    
    Lign_old=Lign_0
    
    def init_old(shape, dtype=None):
        W=[np.zeros(m+1+m*r)]
        return K.constant(W)
    
    dico_func={}
    dico_a={}
    dico_layer_name={}
    for couche in range(0,couches_tot):
        dico_func[couche]=model_old
        dico_layer_name[couche]='last_couche_'+str(couche)
        
        print (dico_layer_name[couche])
        
    
    ##############
    ## Debit iteration Couches  ##
    ##############
        
    for couche in range(0,couches_tot):
        print (" ")
        print ("###############################")
        print ("Couche= ",couche,'/',couches_tot-1)
        print ("intervalles/polynomes= ",m,r)
    
            
    
    
    
        Lign_1 = layers.Dense(m+1, activation=trelu,
                          kernel_initializer=init_W0,
                          use_bias=True, 
                          bias_initializer=init_b0,
                          trainable=False)(inputs)
         
    #    print("couche= ", couche, "FIN Lign_1")
        
        Lign_comp=Lign_1
    
        start_d = dt.datetime.now()
        
    
    
        for m_count in range(0,m):
            for r_count in range(0,r):
    #                print ("Assemblage :  ",m_count,r_count)
                    def init_W3(shape, dtype=None):
                        W=np.array([[m]])
                        return K.constant(W)
                    def init_b3(shape, dtype=None):
                        b=np.array([-m_count])
                        return K.constant(b)        
                    Lign_2 = layers.Dense(1, activation=trelu,
                                          kernel_initializer=init_W3,
                                          bias_initializer=init_b3,
                                          trainable=False)(inputs)
                    def init_W4(shape, dtype=None):
    #                    W=np.array([[1./r]])
                        W=np.array([[(1+1./r)/2.]])
                        return K.constant(W)
                    def init_b4(shape, dtype=None):
    #                    b=np.array([r_count*1./r])
                        b=np.array([r_count*1./r/2.])
                        return K.constant(b)       
                    Lign_2 = layers.Dense(1, activation=trelu,
                                          kernel_initializer=init_W4,
                                          bias_initializer=init_b4,
                                          trainable=False)(Lign_2)
                    
                    start_d5 = dt.datetime.now()
                    Lign_2= model_old(Lign_2)
                    end_d5 = dt.datetime.now()
                    print("  assemb indiv {} seconds".format(end_d5 - start_d5))
                        
                    if (m_count==0) and (r_count==0):
                        Lign_int=Lign_2
                    else: 
                        Lign_int=layers.concatenate([Lign_int, Lign_2])
             
    #    if (couche==1):
    #        sys.exit(0)
            
        # c=m*r
        # Lign_int = layers.Reshape((c,1))(Lign_int)
        # #Lign_int= model_old(Lign_int)
        # Lign_int = layers.Reshape((1,c))(Lign_int)
        # Lign_int = layers.Flatten()(Lign_int)
        
        
        
        #sys.exit(0)
                    
     
        # if (cas==2): ##spec cas_test=2, on ajoute une fonction de base
        #     def init_W3(shape, dtype=None):
        #                 W=np.array([[-m]])
        #                 return K.constant(W)
        #     def init_b3(shape, dtype=None):
        #                 b=np.array([m])
        #                 return K.constant(b)
        #     Lign_2 = layers.Dense(1, activation=trelu,
        #                                   kernel_initializer=init_W3,
        #                                   bias_initializer=init_b3,
        #                                   trainable=False)(inputs)
        #     def init_W4(shape, dtype=None):
        #                 W=np.array([[1./r]])
        #                 return K.constant(W)
        #     def init_b4(shape, dtype=None):
        #                 b=np.array([r_count*1./r])
        #                 return K.constant(b)
        #     Lign_2 = layers.Dense(1, activation=trelu,
        #                                   kernel_initializer=init_W4,
        #                                   bias_initializer=init_b4,
        #                                   trainable=False)(Lign_2)
        #     start_d5 = dt.datetime.now()
        #     Lign_2= model_old(Lign_2)
        #     end_d5 = dt.datetime.now()
        #     print("  assemb indiv {} seconds".format(end_d5 - start_d5))

        #     Lign_int=layers.concatenate([Lign_int, Lign_2])
        #     #Lign_int.trainable=False
        
    #     if (couche==-1):    
    #         Lign_int = layers.Reshape([c,1])(Lign_int)
    #         Lign_int= model_old(Lign_int)
    # #        Lign_int = layers.Reshape((c,1))(Lign_int)
    #         Lign_int = layers.Flatten()(Lign_int)
        
        if (couche==0):    
            Lign_comp=Lign_1
        else:
            Lign_comp=layers.concatenate([Lign_1, Lign_int])
        #Lign_comp = layers.Flatten()(Lign_comp)
    
        Lign_comp.trainable=False
        
     
        
                    
        end_d = dt.datetime.now()
        print("Temps assemblage : {} seconds".format(end_d - start_d))
    
        
        Lign_sortie=layers.Dense(1, activation='linear',use_bias=False, 
                                 kernel_initializer=init_W_old,
     #                            kernel_initializer='random_uniform',
                                 name='sortie_new',
                                 trainable=True)(Lign_comp)
        
    
        #model_new = keras.Model(inputs=inputs, 
        #                        outputs=Lign_sortie)
        
        # start_d = dt.datetime.now()
        # y_sortie=model_new.predict(x_train, verbose=True)
        # end_d = dt.datetime.now()
        # print("Temps evaluation: {} seconds".format(end_d - start_d))
        
        #model_new.summary()
    
        lrate=0.01
        if (couche>0):
            lrate=0.001
        if (couche>1):
            lrate=0.001
        if (couche>2):
            lrate=0.0001
        if (couche>3):
            lrate=0.0001
        if (couche>4):
            lrate=0.00001
        
        #lrate=0.001

       
    #     start_d = dt.datetime.now()
    #     model_new.compile(optimizer=keras.optimizers.Adam(learning_rate=lrate),
    # #               verbose=2,
    #                loss=customLoss)
        
        
    #     end_d = dt.datetime.now()
    #     print("Temps compile: {} seconds".format(end_d - start_d))
    
        
        if (couche==0):
                epochs=100
        else:
                epochs=100
        #epochs=0
        
    
    
        print("Debut boucle couteuse")    
        start_d = dt.datetime.now()    
        model_comp = keras.Model(inputs=inputs, 
                                outputs=Lign_comp)
        
        # print("Debut compilation")
        # model_comp.compile(optimizer=keras.optimizers.Adam(learning_rate=lrate),
        #                                      loss=customLoss)
        # print("Fin compilation")

        print("Debut construction données",x_imp.shape,x_test.shape,x_train.shape)  
        start_d2 = dt.datetime.now()
        print("x_imp")
        #start_d3 = dt.datetime.now()
        #xxx_p_comp=model_comp.predict(x_imp,verbose=1)
        #end_d3 = dt.datetime.now()
        #print("  mes 1 {} seconds".format(end_d3 - start_d3))
        
        start_d4 = dt.datetime.now()
        xxx_p_comp=model_comp(x_imp)
        
        encore=model_comp(tf.convert_to_tensor(x_p))
        end_d4 = dt.datetime.now()
        print("  mes 2 {} seconds".format(end_d4 - start_d4))

        
        print("x_test, x_train")
        #x_comp_test=model_comp.predict(x_test,verbose=1)
        x_comp_test=model_comp(x_test)
        #x_comp=model_comp.predict(x_train,verbose=1)
        x_comp=model_comp(x_train)
        end_d2 = dt.datetime.now()
        print("  Temps donn:ées {} seconds".format(end_d2 - start_d2))
        print("Fin construction données")    
        
    
        if (couche==0):
            b=m+1
        else:
            b=m+1+m*r
            if (cas==2):
                b=6
            
        inputs_add_comp = keras.Input(shape=(b),
                                          name="inputs_add_comp")
        Lign_add_comp=layers.Dense(1, activation='linear',use_bias=False, 
                                 kernel_initializer=init_W_old,
                                 name=dico_layer_name[couche],
                                 trainable=True)(inputs_add_comp)
    
        print("Debut construction model")    
        model_add_comp = keras.Model(inputs=inputs_add_comp, 
                                outputs=Lign_add_comp)
        model_add_comp.summary()
        print("Fin construction model")    
    
        model_add_comp.compile(#optimizer=keras.optimizers.SGD(learning_rate=lrate),
                   optimizer=keras.optimizers.Adam(learning_rate=lrate),
                   #verbose=0,
                   loss=customLoss)
        print ("Fit avec epochs/lrate= ", epochs,", ",lrate)
        model_add_comp.fit(x_comp, y_train,
    		  batch_size=128,
    		  epochs=200,
    #         callbacks=[history,historyepoch,historyval,lrate,historylearnrate],
    		  verbose=0,callbacks=[TqdmCallback(verbose=0)],
    #		  verbose=2,
              validation_data=(x_comp_test, y_test)
    		  )    
        
        z0=model_add_comp.predict(xxx_p_comp,verbose=1) 


        Lign_final=model_add_comp(Lign_comp)
        model_new=keras.Model(inputs=inputs, 
                                outputs=Lign_final)
        model_new.trainable = False
        model_old=model_new
        dico_func[couche]=model_old
        
        #model_old.summary()
        
        end_d = dt.datetime.now()
        print("  Temps boucle couteuse : {} seconds".format(end_d - start_d))    
        
        W_new=model_add_comp.get_weights()
        #print ("Weights=",W_new)
        
        dico={}
        for layer in model_add_comp.layers:
                dico[layer.name]=np.array(layer.get_weights(),
                                          dtype=object)
                

             
        K_contract=0
        a=dico[dico_layer_name[couche]]
        print ("a=",a)

        if (couche>0):
            for i in range(m+1,m+1+m*r):
                K_contract=K_contract+abs(a[0][i][0])
            print ("Constante K=",K_contract)
            #print(a)
        
    # y_couche_der[0]=(y_couche[1] -y_couche[0])*n
    #     for i in range(1,n-1):
    #             y_couche_der[i]=(y_couche[i+1] -y_couche[i-1])*n/2.
    #     y_couche_der[n-1]=(y_couche[n-1] -y_couche[n-2])*n
    
    
        def init_W_old(shape, dtype=None):
                 a=dico[dico_layer_name[couche-1]]
                 b=m+1+m*r
                 W=[np.zeros(b)]
                 
                 #print("COUCOUC",couche,a)
                 if (couche>1):
                      for i in range(0,b):
                              W[0][i]=a[0][i][0]
                 else:
                      for i in range(0,m+1):
                              W[0][i]=a[0][i][0]
                      for i in range(m+1,b):        
                              W[0][i]=0
                             
                 # W[0][0]=0
                 # W[0][1]=0
                 # W[0][2]=0
                 # W[0][3]=0
                 # W[0][4]=0
                 # W[0][5]=0
                 # W[0][6]=1
                 
                 Z=np.transpose(W)
                 return K.constant(Z)    
        
    #########################
    #  Fin itération Couches 
    #########################            
    
    
        print("Impression")
    
        #z0=model_new.predict(xxx_p,verbose=1) 
        z_new=np.linspace(0,1,n)
        z_new_der=np.linspace(0,1,n)
        
        


        
        for i in range(0,n):
                z_new[i]=z0[i,0]
                
                z_new_der[0]=(z_new[1] -z_new[0])*n
                for i in range(1,n-1):
                    z_new_der[i]=(z_new[i+1] -z_new[i-1])*n/2
                    z_new_der[n-1]=(z_new[n-1] -z_new[n-2])*n
                    
        # print ("x_imp",x_imp)
        # print ("z0",z0)
        # print ("znew",z_new)
        # sys.exit(0)
                    
    #plt.plot(x_p,z_new,label='ref')
    #plt.legend(loc="upper right")
    #plt.show()
    
    
            
        string_file='%5.8f %5.8f %5.8f \n'
        
        strstr_train='./Sortie_'+str(couche)+'.txt'
        fichier_sortie = open(strstr_train,'w')
        
        for i in range(0,n):
            a=x_p[i]
            #    b=y_couche[i]
            c=z_new[i]
            #    d=y_couche_der[i]
            e=z_new_der[i]
            fichier_sortie.write(string_file %  (a,c,e) )
        
        fichier_sortie.close()
    
    #string_file='%5.8f %5.8f \n'
    
    
    
    end = dt.datetime.now()
    #print("Training duration: {} seconds".format(end - start))
    

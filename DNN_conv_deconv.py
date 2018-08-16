
# coding: utf-8

# In[12]:


import tensorflow as tf
import keras
import numpy as np 
import keras.models
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D,Conv2DTranspose
from keras import backend as K
from math import log
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras import losses
from keras import regularizers
# In[13]:

import os
import tensorflow as tf


os.environ["CUDA VISIBLE DEVICES" ] = "3"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7
K.set_session(tf.Session(config = config))

def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


def get_next_character(f):
    """Reads one character from the given textfile"""
    c = f.read(1)
    while c:
        yield c
        c = f.read(1)
num_train_examples = 24004 #24004
num_test_examples = 4800 #4800
num_valid_examples = 3999  #3999

def load_data(kind,label):
    #this function loads data from the
    #files & returns numpy arrays of test and training examples
    '''ly = []
        for i in range(0,256):
        ly.append(i)
        
        a = np.array(ly)
        
        b = np.zeros((256, 256))
        b[np.arange(256), a] = 1'''
    
    x_train = []
    y_train = []
    x_valid = []
    y_valid = []
    x_test = []
    y_test = []
    
    
    
    if(kind == "train"):
        num_examples = num_train_examples
        #num_examples = [500,500,500,500]
        folder_name = "4095_randomized/training/"
        out_folder_name = "4095_noisy_randomized_0p01/training/"
    if(kind == "valid"):
        num_examples = num_valid_examples
        #num_examples = [500,500,500,500]
        folder_name = "4095_randomized/validation/"
        out_folder_name = "4095_noisy_randomized_0p01/validation/"
    if(kind == "test"):
        num_examples = num_test_examples
        #num_examples = [500,500,500,500]
        folder_name = "4095_randomized/testing/"
        out_folder_name = "4095_noisy_randomized_0p01/testing/"
    for i in range(num_examples):
        if i % 500 == 0:
            print("Loading "+kind+"  example: "+str(i)+" for label : "+label)
        with open(folder_name+label+"/"+str(i)+".txt", 'rb') as f:
                count  = 0
                l = []
                for c in get_next_character(f) :
            
                    l.append(int(c))
                    count = count+1
                l = np.array(l)
                #label_counter[j] = label_counter[j]+1
                if(kind == "train"):
                    y_train.append(l)
                
                if(kind == "test"):
                    y_test.append(l)
                
                if(kind == "valid"):
                    y_valid.append(l)

        with open(out_folder_name+label+"/"+str(i)+".txt", 'rb') as f:
                count  = 0
                l = []
                for c in get_next_character(f) :
                
                    l.append(int(c))
                    count = count+1
                
                l = np.array(l)
                #label_counter[j] = label_counter[j]+1
                if(kind == "train"):
                    x_train.append(l)

                if(kind == "test"):
                    x_test.append(l)
        
                if(kind == "valid"):
                    x_valid.append(l)

                del l


    if(kind == "train"):
        return (np.array(x_train), np.array(y_train))
    if(kind == "valid"):
        return (np.array(x_valid), np.array(y_valid))
    if(kind == "test"):
        return (np.array(x_test), np.array(y_test))
        





# In[14]:


def negGrowthRateLoss(b,q):
    return (K.mean(-K.log(b +pow(-1,b)+pow(-1,b+1)*q)/K.log(2.0)))


# In[15]:

def runProgram():
    k = 4095
    img_rows, img_cols = k, 1 #not one hot
    # the data, shuffled and split between train and test sets

    (x_train, y_train) = load_data("train","jpeg")
    (x_valid, y_valid) = load_data("valid","jpeg")
    (x_test, y_test) = load_data("test","jpeg")
    print('Before reshape:')
    print('x_train shape:', x_train.shape)
    print('x_valid shape:', x_valid.shape)
    print('x_test shape:', x_test.shape)
    x_train = np.reshape(x_train,(len(x_train), 4095,1,1))
    x_valid = np.reshape(x_valid,(len(x_valid),4095,1,1))
    x_test = np.reshape(x_test,(len(x_test),4095,1,1))
    y_train = np.reshape(y_train,(len(y_train), 4095,1,1))
    y_valid = np.reshape(y_valid,(len(y_valid),4095,1,1))
    y_test = np.reshape(y_test,(len(y_test),4095,1,1))
    print('After reshape:')
    print('x_train shape:', x_train.shape)
    print('x_valid shape:', x_valid.shape)
    print('x_test shape:', x_test.shape)


    input_shape = (img_rows,img_cols,1)



    # convert class vectors to binary class matrices

    print('Shuffling in unison ')
    shuffle_in_unison(x_train,y_train)
    shuffle_in_unison(x_valid,y_valid)
    shuffle_in_unison(x_test,y_test)
    #print('y_train shape:', y_train.shape)
    #print(y_train.shape[0], 'train labels')
    #print(y_test.shape[0], 'test labels')

    batch_size = 50
    epochs = 20


    input_img = Input(shape=input_shape) # adapt this if using `channels_first` image data format
  
    
    x = Conv2D(100, (24,1), activation='relu')(input_img)
    #x = Dropout(0.5)(x)
    x = Conv2D(200, (16, 1), activation='relu')(x)
    #x = Dropout(0.5)(x)
    #x = Conv2D(100, (8, 1), activation='relu')(x)
    #x = Dropout(0.5)(x)
    #x = Conv2D(300, (3, 1), activation='relu')(x)
    #x = Conv2D(64, (4, 1), activation='relu')(x)
    #x = Conv2D(64, (4, 1), activation='relu')(x)
    #x = Conv2D(64, (4, 1), activation='relu')(x)
    #x = Conv2D(64, (4, 1), activation='relu')(x)
    encoded = Conv2D(40, (8, 1), activation='relu')(x)
    

    x = Conv2DTranspose(200, (8, 1), activation='relu')(encoded)
   #x = Dropout(0.5)(x)
    x = Conv2DTranspose(100, (16, 1), activation='relu')(x)
   #x = Dropout(0.5)(x)
    #x = Conv2DTranspose(200, (8, 1), activation='relu')(x)
    #x = Dropout(0.5)(x)
    #x = Conv2DTranspose(100, (16, 1), activation='relu')(x)
    #x = Conv2DTranspose(64, (4, 1), activation='relu')(x)
    #x = Conv2DTranspose(32, (4, 1), activation='relu')(x)
    #x = Conv2DTranspose(32, (8, 1), activation='relu')(x)
    #x = Conv2DTranspose(320, (16, 1), activation='relu')(x)
    decoded = Conv2DTranspose(1, (24, 1), activation='sigmoid')(x)
    

    model = Model(input_img, decoded)
    model.compile(loss=negGrowthRateLoss,optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
    filepath = 'auto_weights_jpeg_10_0p01.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    csv_logger = CSVLogger('auto_losses_jpeg_10_0p01.csv')
    plot_model(model,to_file='auto_model_jpeg_10_0p01.png',show_shapes =  True)
    #Let's train it for 100 epochs:
    #model.load_weights('auto_weights_jpeg_11_0p01.h5')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_valid, y_valid),callbacks=[checkpoint,csv_logger])






#optimization algorithm, metrics, and loss functions


 #train

#model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1)



    model.save('auto_model_jpeg_10_0p01.h5')
    model.load_weights(filepath)
    predictions = model.predict(x_test,verbose=0)
    print(predictions.shape)
    model.summary()
    p_p_y = np.array([[0.0,0.0],[0.0,0.0]])
    ct = np.array([0.0,0.0])
    p_p = np.array([0.0,0.0])
    p_y = np.array([0.0,0.0])
    for j in range(0,num_test_examples):
        #filename = open("results/res_test_exmpl_"+str(j)+".csv",'w')
        for i in range(0,4095):
            
            #filename.write(str(predictions[j][i])+","+str(y_test[j][i])+","+str(x_test[j][i])+"\n")
            #if(y_test[j][i][0][0] != x_test[j][i][0][0]):
            #print(str(y_test[j][i][0][0])+" "+str(x_test[j][i][0][0])+"  "+str(predictions[j][i][0][0]))
            if( y_test[j][i] == 0):
                p_p_y[1][0] = p_p_y[1][0]+predictions[j][i][0][0]
            #p[1][y_test[j][i]] = p[1][y_test[j][i]]+(1.0-predictions[j][i])
                ct[0] = ct[0]+1.0
            else:
                p_p_y[1][1] = p_p_y[1][1]+predictions[j][i][0][0]
            #p[0][y_test[j][i]] = p[0][y_test[j][i]]+(1.0-predictions[j][i])
                ct[1] = ct[1]+1.0
            p_p[1] = p_p[1]+predictions[j][i][0][0]

        #filename.close()

    p_p_y[1][0] = p_p_y[1][0]/ct[0]
    p_p_y[1][1] = p_p_y[1][1]/ct[1]

    p_p_y[0][0] = 1.0 - p_p_y[1][0]
    p_p_y[0][1] = 1.0 - p_p_y[1][1]


    p_p[1] = p_p[1]/(ct[0]+ct[1])
    p_p[0] = 1-p_p[1]

    p_y[0] = ct[0]/(ct[0]+ct[1])
    p_y[1] = ct[1]/(ct[0]+ct[1])

    mut_inf = 0.0

    for i in range(0,2):
        for j in range(0,2):
            p_p_y[i][j] = p_p_y[i][j]*p_y[j]
    

    #ct = ct/sum(ct)
    #ct_pred = ct_pred/sum(ct_pred)

    file = open("auto_results_jpeg_10_0p01.txt",'w+')
    file.write("Joint:\n")
    file.write(str(p_p_y))
    #print(sum(p_p_y))
    file.write("\n Marginal: Predictions:\n")
    file.write(str(p_p))
    #print(sum(p_p))
    file.write("\n Marginal: Labels:\n")
    file.write(str(p_y))


    '''
    for i in range(0,2):
        for j in range(0,2):
            p[i][j] = p[i][j]*ct[j]
    '''


    for i in range(0,2):
        for j in range(0,2):
            if(p_p_y[i][j]!=0):
                mut_inf = mut_inf+(p_p_y[i][j]*log((p_p_y[i][j])/(p_p[i]*p_y[j]))/log(2.0))


    file.write("\nMutual information:\n")
    file.write(str(mut_inf))


runProgram()



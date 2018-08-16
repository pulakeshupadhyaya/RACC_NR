#STORES PROBABILITIES FOR LDPC DECODING
from __future__ import print_function
import keras
from keras.models import load_model
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D,Conv2DTranspose
from keras import backend as K
import numpy as np
from random import randint
#from keras.layers.convolutional import Convolution2D
from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard
from time import time 
import gc

from keras.utils import plot_model
from keras import Model
#import keras.losses
#keras.losses.custom_loss = custom_loss

batch_size = 50
num_classes = 4
epochs = 15
num_test_examples = 4800


import os
import tensorflow as tf
os.environ["CUDA VISIBLE DEVICES" ] = "2,3"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.8
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
        out_folder_name = "4095_noisy_randomized_0p008/training/"
    if(kind == "valid"):
        num_examples = num_valid_examples
        #num_examples = [500,500,500,500]
        folder_name = "4095_randomized/validation/"
        out_folder_name = "4095_noisy_randomized_0p008/validation/"
    if(kind == "test"):
        num_examples = num_test_examples
        #num_examples = [500,500,500,500]
        folder_name = "4095_randomized/testing/"
        out_folder_name = "4095_noisy_randomized_0p008/testing/"
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

def negGrowthRateLoss(b,q):
    return (K.mean(-K.log(b +pow(-1,b)+pow(-1,b+1)*q)/K.log(2.0)))

k = 4095
img_rows, img_cols = k, 1 #not one hot
# the data, shuffled and split between train and test sets
#(x_train, y_train) = load_data("train","html")
#(x_valid, y_valid) = load_data("valid","html")
(x_test, y_test) = load_data("test","html")

#x_train = x_train.reshape(x_train.shape[0], img_rows,img_cols,1)
#x_valid = x_valid.reshape(x_valid.shape[0], img_rows,img_cols,1)
x_test = x_test.reshape(x_test.shape[0],img_rows,img_cols,1)
'''
    y_train = y_train.reshape(y_train.shape[0], k)
    y_valid = y_valid.reshape(y_valid.shape[0], k)
    y_test = y_test.reshape(y_test.shape[0],k)
    '''
#print('After reshape:')
#print('x_train shape:', x_train.shape)
#print('x_valid shape:', x_valid.shape)
print('x_test shape:', x_test.shape)

input_shape = (img_rows, img_cols,1)



# convert class vectors to binary class matrices

#print('Shuffling in unison ')
#shuffle_in_unison(x_train,y_train)
#shuffle_in_unison(x_valid,y_valid)
#shuffle_in_unison(x_test,y_test)






    
input_img = Input(shape=input_shape) # adapt this if using `channels_first` image data format
    
    
x = Conv2D(100, (8,1), activation='relu')(input_img)
x = Conv2D(200, (3, 1), activation='relu')(x)
x = Conv2D(300, (3, 1), activation='relu')(x)
encoded = Conv2D(40, (3, 1), activation='relu')(x)
    
x = Conv2DTranspose(300, (3, 1), activation='relu')(encoded)
x = Conv2DTranspose(200, (3, 1), activation='relu')(x)
x = Conv2DTranspose(100, (3, 1), activation='relu')(x)
decoded = Conv2DTranspose(1, (8, 1), activation='sigmoid')(x)
    
    
model = Model(input_img, decoded)
model.compile(loss=negGrowthRateLoss,optimizer=keras.optimizers.Adam())



model.load_weights('auto_weights_4_0p008.h5')
predictions = model.predict(x_test,verbose=0)
    #print(predictions.shape)

for j in range(0,num_test_examples):
    filename = open("results/0p008/"+str(j)+".csv",'w')
    for i in range(0,4095):
            
        filename.write(str(predictions[j][i][0][0])+"\n")


filename.close()


'''Trains a simple convnet on the file types data set
    
    Gets to 90%+ test accuracy
    
    Not optimal in terms of memory : reads too much into memory
    Possible improvement : Read one batch at a time from the storage
    '''

from __future__ import print_function
import keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D,Conv2DTranspose
from keras import backend as K
import numpy as np
from random import randint
#from keras.layers.convolutional import Convolution2D

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

def load_FTR_data(label,i):
    start_time = time()
    x_test = []
    y_test = []
   
    if(label == 'html'):
        j = 0
    if(label == 'jpeg'):
        j = 1
    if(label == 'pdf'):
        j = 2
    if(label == 'latex'):
        j = 3
    folder_name = "4095_noisy_randomized_0p02/testing/"
    with open(folder_name+label+"/"+str(i)+".txt", 'rb') as f:
        l = []
        for c in get_next_character(f) :
                #print(int(c))
            l.append(int(c))

        l = np.array(l)
        x_test.append(l)
        y_test.append(j)
	del l
    #print("FTR_data :"+str(time()-start_time))
    return (np.array(x_test), np.array(y_test))

def load_CM_data(label, i):
    start_time = time()
    x_test = []
    y_test = []
    folder_name = "4095_randomized/testing/"
    out_folder_name = "4095_noisy_randomized_0p02/testing/"
    
    with open(folder_name+label+"/"+str(i)+".txt", 'rb') as f:
        l = []
        for c in get_next_character(f) :
            l.append(int(c))
    
        l = np.array(l)
        y_test.append(l)
        del l    
        
    
    with open(out_folder_name+label+"/"+str(i)+".txt", 'rb') as f:
        count  = 0
        l = []
        for c in get_next_character(f) :
                
            l.append(int(c))
            count = count+1
        
        l = np.array(l)
            #label_counter[j] = label_counter[j]+1
        x_test.append(l)
        del l


    gc.collect()
    #print("CModel :"+str(time()-start_time))
    return (np.array(x_test), np.array(y_test))






def negGrowthRateLoss(b,q):
    return (K.mean(-K.log(b +pow(-1,b)+pow(-1,b+1)*q)/K.log(2.0)))





def DNN_model_load(pred):
    start_time = time()
    img_rows, img_cols = 4095, 1 #not one hot
    input_shape = (img_rows, img_cols,1)
    input_img = Input(shape=input_shape)
    if (pred == 0):
         # adapt this if using `channels_first` image data format
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


        model.load_weights('auto_weights_4_0p016.h5')
    if (pred == 1):
         # adapt this if using `channels_first` image data format
        x = Conv2D(100, (24,1), activation='relu')(input_img)
        x = Conv2D(100, (16, 1), activation='relu')(x)
        x = Conv2D(100, (8, 1), activation='relu')(x)
        encoded = Conv2D(80, (8, 1), activation='relu')(x)
        
        x = Conv2DTranspose(100, (8, 1), activation='relu')(encoded)
        x = Conv2DTranspose(100, (8, 1), activation='relu')(x)
        x = Conv2DTranspose(100, (16, 1), activation='relu')(x)
        decoded = Conv2DTranspose(1, (24, 1), activation='sigmoid')(x)
        
        
        model = Model(input_img, decoded)
        model.compile(loss=negGrowthRateLoss,optimizer=keras.optimizers.Adam())
        
        
        model.load_weights('auto_weights_jpeg_11_0p016.h5')

    if(pred == 2):
        # adapt this if using `channels_first` image data format
        x = Conv2D(100, (24,1), activation='relu')(input_img)
        x = Conv2D(200, (16, 1), activation='relu')(x)
        x = Conv2D(300, (8, 1), activation='relu')(x)
        encoded = Conv2D(40, (8, 1), activation='relu')(x)
        
        x = Conv2DTranspose(300, (8, 1), activation='relu')(encoded)
        x = Conv2DTranspose(200, (8, 1), activation='relu')(x)
        x = Conv2DTranspose(100, (16, 1), activation='relu')(x)
        decoded = Conv2DTranspose(1, (24, 1), activation='sigmoid')(x)
        
        
        model = Model(input_img, decoded)
        model.compile(loss=negGrowthRateLoss,optimizer=keras.optimizers.Adam())
        
        
        model.load_weights('auto_weights_pdf_8_0p016.h5')
    if(pred == 3):
        x = Conv2D(100, (8,1), activation='relu')(input_img)
        x = Conv2D(200, (3, 1), activation='relu')(x)
        x = Conv2D(300, (3, 1), activation='relu')(x)
        x = Conv2D(300, (3, 1), activation='relu')(x)
        encoded = Conv2D(40, (3, 1), activation='relu')(x)
        
        x = Conv2DTranspose(300, (3, 1), activation='relu')(encoded)
        x = Conv2DTranspose(300, (3, 1), activation='relu')(x)
        x = Conv2DTranspose(200, (3, 1), activation='relu')(x)
        x = Conv2DTranspose(100, (3, 1), activation='relu')(x)
        
        decoded = Conv2DTranspose(1, (8, 1), activation='sigmoid')(x)
        
        
        model = Model(input_img, decoded)
        model.compile(loss=negGrowthRateLoss,optimizer=keras.optimizers.Adam())
        
        
        model.load_weights('auto_weights_latex_6_0p016.h5')
    gc.collect()
    #print("DNNModel :"+str(time()-start_time))
    return model

def FTR(label,i):
    start_time = time()
    img_rows, img_cols = 4095, 1
    model = load_model('4095_0p016_my_model_9layer_0drop.h5')
    model.load_weights('4095_0p016_my_model_weights_9layer_0drop.h5')
    (x_test,y_test) = load_FTR_data(label,i)
    x_test = x_test.reshape(x_test.shape[0],img_rows, img_cols,1)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    predictions = model.predict(x_test,verbose=0)
    pred = np.argmax(predictions)
    gc.collect()
    del x_test
    del y_test
    #print("FTR :"+str(time()-start_time))
    return pred

labels = ["html","jpeg","pdf","latex"]
filename = None
def store_DNN_values(label,i):
    start_time = time()
    pred = FTR(label,i)
    model = DNN_model_load(pred)
    #model.summary()
    print(str(i)+" : "+label+" : "+labels[pred])
    k = 4095
    img_rows, img_cols = k, 1 #not one hot
    (x_test, y_test) = load_CM_data(label,i)
    x_test = x_test.reshape(x_test.shape[0],img_rows,img_cols,1)
    predictions = model.predict(x_test,verbose=0)
    print(predictions.shape)
    del x_test
    del y_test    
    filename = open("results_0p016/"+label+"/0p02/"+str(i)+".csv",'w')
    for k in range(0,4095):
        filename.write(str(predictions[0][k][0][0])+"\n")
    filename.close()
    
    gc.collect()
    print("Store DNN :"+str(time()-start_time))
import sys
def main():
     i = int(sys.argv[1])
     label = str(sys.argv[2])
     store_DNN_values(label,i)
     gc.collect()
main()

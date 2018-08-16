'''Trains a simple convnet on the file types data set

Gets to 90%+ test accuracy

Not optimal in terms of memory : reads too much into memory
Possible improvement : Read one batch at a time from the storage
'''

from __future__ import print_function
import keras
from keras.models import load_model
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
from random import randint
#from keras.layers.convolutional import Convolution2D
from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard
from time import time 
import gc
from keras.utils import plot_model
batch_size = 100
num_classes = 4
epochs = 30

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
    
    
    label_list = ['html','jpeg','pdf','latex'] #list of labels
    #label_onehot = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    label_counter = [ 1, 1, 1, 1] #read a particular example
    if(kind == "train"):
        num_examples = [24004,24004,24004,24004]
        folder_name = "4095_noisy_randomized_0p014/training/"
    if(kind == "valid"):
        num_examples = [3999,3999,3999,3999]
        folder_name = "4095_noisy_randomized_0p014/validation/"
    if(kind == "test"):
        num_examples = [4800,4800,4800,4800]
        folder_name = "4095_noisy_randomized_0p01/testing/"
    if(label == 'html'):
    	j = 0
    if(label == 'jpeg'):
	j = 1
    if(label == 'pdf'):
	j= 2
    if(label == 'latex'):
	j= 3
    for i in range(num_examples[j]):
        if i % 500 == 0:
            print("Loading "+kind+"  example: "+str(i)+" for label : "+label)
        with open(folder_name+label+"/"+str(i)+".txt", 'rb') as f:
            count  = 0
            l = []
            for c in get_next_character(f) :
            #print(int(c))
                l.append(int(c))
                count = count+1
            l = np.array(l)
                #label_counter[j] = label_counter[j]+1
            if(kind == "train"):
                x_train.append(l)
                y_train.append(j)
            if(kind == "test"):
                x_test.append(l)
                y_test.append(j)
            if(kind == "valid"):
                x_valid.append(l)
                y_valid.append(j)
            del l


    if(kind == "train"):
    	return (np.array(x_train), np.array(y_train))
    if(kind == "valid"):
        return (np.array(x_valid), np.array(y_valid))
    if(kind == "test"):
        return (np.array(x_test), np.array(y_test))

#read shuffled data and split between train and test sets


# input image dimensions
#img_rows, img_cols = 8192, 256 #one hot
img_rows, img_cols = 4095, 1 #not one hot
# the data, shuffled and split between train and test sets

(x_test, y_test) = load_data("test","html")


x_test = x_test.reshape(x_test.shape[0],img_rows, img_cols,1)

input_shape = (img_rows, img_cols,1)




# convert class vectors to binary class matrices

y_test = keras.utils.to_categorical(y_test, num_classes)





model = Sequential()
model = load_model('4095_0p01_my_model_9layer_0drop.h5')
model.load_weights('4095_0p01_my_model_weights_9layer_0drop.h5')
score = model.evaluate(x_test, y_test, verbose=0) #test

print('Test loss:', score[0])
print('Test accuracy:', score[1])


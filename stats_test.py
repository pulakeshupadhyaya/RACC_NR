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
from keras.layers.convolutional import Convolution2D
from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard
from time import time 
import gc
from keras.utils import plot_model
from math import floor
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


def load_data(kind):
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
        num_examples = [1000,1000,1000,1000]
        #num_examples = [15087,15089,15088,15027]
        folder_name = "randomized/training/"
    if(kind == "valid"):
        num_examples = [2514,2525,2514,2504]
        #num_examples = [20,20,20,20]
        folder_name = "randomized/validation/"
    if(kind == "test"):
        #num_examples = [20,20,20,20]
        num_examples = [3398,3385,3397,3458]
        folder_name = "randomized/testing/"
    j = 0
    for label in label_list:
        print(j)
        for i in range(num_examples[j]):
            if i % 500 == 0:
                print("Loading "+kind+"  example: "+str(i)+" for label : "+label)
            with open(folder_name+label+"/"+str(i)+".txt", 'rb') as f:
                count  = 0
                l = []
                for c in get_next_character(f) :
                        
                    #l.append(b[ord(c)]) #one hot
                    l.append(ord(c)*1.0/255.0)
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
        j = j+1
    if(kind == "train"):
        return (np.array(x_train), np.array(y_train))
    if(kind == "valid"):
        return (np.array(x_valid), np.array(y_valid))
    if(kind == "test"):
        return (np.array(x_test), np.array(y_test))

#read shuffled data and split between train and test sets 
    
    
# input image dimensions
#img_rows, img_cols = 8192, 256 #one hot
img_rows, img_cols = 8192, 1 #not one hot
# the data, shuffled and split between train and test sets
(x_train, y_train) = load_data("train")
#(x_valid, y_valid) = load_data("valid")
#(x_test, y_test) = load_data("test")
#print('Before reshape:')
#print('x_train shape:', x_train.shape)
#print('x_valid shape:', x_valid.shape)
#print('x_test shape:', x_test.shape)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols,1)
#x_valid = x_valid.reshape(x_valid.shape[0], img_rows, img_cols,1)
#x_test = x_test.reshape(x_test.shape[0],img_rows, img_cols,1)

input_shape = (img_rows, img_cols,1)


#print(x_train.shape[0], 'train samples')
#print(x_valid.shape[0], 'train samples')
#print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_valid = keras.utils.to_categorical(y_valid, num_classes)
#y_test = keras.utils.to_categorical(y_train, num_classes)
#print('Shuffling in unison ')
#shuffle_in_unison(x_train,y_train)
#shuffle_in_unison(x_valid,y_valid)
#shuffle_in_unison(x_test,y_test)
#print('y_train shape:', y_train.shape)
#print(y_train.shape[0], 'train labels')
#print(y_train.shape[0], 'test labels')

def get_activations(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], model.layers[layer].output)
    activations = get_activations([X_batch,0])
    return activations

model = Sequential()
model = load_model('my_model_3layer_0drop.h5')
model.load_weights('my_model_weights_3layer_0drop.h5')

def get_activations(model, layer, X):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output,])
    activations = get_activations([X,0])
    return activations

nex = [1000,1000,1000,1000]
#nex =  [15087,15089,15088,15027]
llst = ['html','jpeg','pdf','latex']
#counters = [[[0]*10001]*64]*4
counters = np.zeros((4,64,101))
norm = np.zeros((4,64))
seen_examples = 0
for k in range(0,4):
    
    for i in range(seen_examples,seen_examples+nex[k]):
        seen_examples = seen_examples+1
        if(i % 100 == 1):
            print("Train example : "+str(i)+"/60291")
        f = get_activations(model,5,x_train[i:i+1])
        
        
        l = f[0]
        count = 0
        
        for ctr in range(0,64):
            for j in range(0,1022):
                box = min(100,int(floor(l[0][j][0][ctr]*100)))
                #ccc[k][ctr] = ccc[k][ctr]+1
                counters[k][ctr][box] =  counters[k][ctr][box]+1
                norm[k][ctr] = norm[k][ctr]+1

for ctr in range(0,64):
    for k in range(0,4):
        for box in range(0,101):
            print(norm[k][ctr])
            counters[k][ctr][box] = counters[k][ctr][box]/norm[k][ctr]

print(counters[0][0])
for ctr in range(0,64):
    
        file = open("feature_"+str(ctr)+".txt","w+")
        file.write("frequency,html,jpeg,pdf,latex\n")
        for box  in range(0,101):
            file.write(str(round(box*1.0/100,3))+",")
            for k in range(0,4):
                
                    if k < 3:
                        file.write(str(counters[k][ctr][box])+",")
                    else :
                        file.write(str(counters[k][ctr][box])+"\n")
        
        file.close()

#print(ctr)



#score = model.evaluate(x_test, y_test, verbose=0) #test

#print('Test loss:', score[0])
#print('Test accuracy:', score[1])


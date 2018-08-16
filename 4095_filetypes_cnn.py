

from __future__ import print_function
import keras
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
from keras.callbacks import ModelCheckpoint
from time import time 
import gc
from keras.utils import plot_model
batch_size = 50
num_classes = 4
epochs = 20
import os
import tensorflow as tf


os.environ["CUDA VISIBLE DEVICES" ] = "3"
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
        num_examples = [24004,24004,24004,24004]
        folder_name = "4095_noisy_randomized_0p02/training/"
    if(kind == "valid"):
        num_examples = [3999,3999,3999,3999]
        folder_name = "4095_noisy_randomized_0p02/validation/"
    if(kind == "test"):
        num_examples = [4800,4800,4800,4800]
        folder_name = "4095_noisy_randomized_0p02/testing/"
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
img_rows, img_cols = 4095, 1 #not one hot
# the data, shuffled and split between train and test sets
(x_train, y_train) = load_data("train")
(x_valid, y_valid) = load_data("valid")
(x_test, y_test) = load_data("test")
print('Before reshape:')
print('x_train shape:', x_train.shape)
print('x_valid shape:', x_valid.shape)
print('x_test shape:', x_test.shape)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols,1)
x_valid = x_valid.reshape(x_valid.shape[0], img_rows, img_cols,1)
x_test = x_test.reshape(x_test.shape[0],img_rows, img_cols,1)

input_shape = (img_rows, img_cols,1)


print(x_train.shape[0], 'train samples')
print(x_valid.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print('Shuffling in unison ')
shuffle_in_unison(x_train,y_train)
shuffle_in_unison(x_valid,y_valid)
shuffle_in_unison(x_test,y_test)
print('y_train shape:', y_train.shape)
print(y_train.shape[0], 'train labels')
print(y_test.shape[0], 'test labels')

#batch_size  = 50

model = Sequential()
#model.add(Convolution2D(32, 5,256,
 #                activation='relu',
 #                input_shape=input_shape))
#one hot
model.add(Conv2D(32, (3,1),
              activation='relu',
              input_shape=input_shape))
# 3 x 1 convolution layer
model.add(MaxPooling2D(pool_size=(2, 1), strides=None, padding='valid', data_format=None))
#2 x 1 max pool
model.add(Conv2D(64, (3,1),
                 activation='relu',
                 input_shape=input_shape))
#3 x 1 convolution layer
model.add(MaxPooling2D(pool_size=(2, 1), strides=None, padding='valid', data_format=None))

model.add(Conv2D(64, (3,1),
                 activation='relu',
                 input_shape=input_shape))
#3 x 1 convolution layer
model.add(MaxPooling2D(pool_size=(2, 1), strides=None, padding='valid', data_format=None))

model.add(Conv2D(64, (3,1),
                 activation='relu',input_shape=input_shape))
#3 x 1 convolution layer
model.add(MaxPooling2D(pool_size=(2, 1), strides=None, padding='valid', data_format=None))
model.add(Conv2D(64, (3,1),
                 activation='relu',input_shape=input_shape))
#3 x 1 convolution layer
model.add(MaxPooling2D(pool_size=(2, 1), strides=None, padding='valid', data_format=None))

model.add(Conv2D(64, (3,1),
                 activation='relu',input_shape=input_shape))
#3 x 1 convolution layer
model.add(MaxPooling2D(pool_size=(2, 1), strides=None, padding='valid', data_format=None))

model.add(Conv2D(64, (3,1),
                 activation='relu',input_shape=input_shape))
#3 x 1 convolution layer
model.add(MaxPooling2D(pool_size=(2, 1), strides=None, padding='valid', data_format=None))

model.add(Conv2D(64, (3,1),
                 activation='relu',input_shape=input_shape))
#3 x 1 convolution layer
model.add(MaxPooling2D(pool_size=(2, 1), strides=None, padding='valid', data_format=None))

model.add(Conv2D(64, (3,1),
                 activation='relu',input_shape=input_shape))
#3 x 1 convolution layer
model.add(MaxPooling2D(pool_size=(2, 1), strides=None, padding='valid', data_format=None))
'''
model.add(Conv2D(64, (3,1),
                 activation='relu',input_shape=input_shape))
#3 x 1 convolution layer
model.add(MaxPooling2D(pool_size=(2, 1), strides=None, padding='valid', data_format=None))
'''
#model.add(Dropout(0.75))
model.add(Flatten())
#flatten
model.add(Dense(num_classes,activation='softmax'))
#dense layer with softmax function
model.compile(loss=keras.losses.categorical_crossentropy,
optimizer=keras.optimizers.Adadelta(),
             metrics=['accuracy'])
#optimization algorithm, metrics, and loss functions
plot_model(model,to_file='4095_p=0p020_model_9lay_0drop.png',show_shapes =  True)


tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
#log data to tensorboard


csv_logger = CSVLogger('4095_p=0p020_results_9lay_0drop.csv')
#log accuracy and loss results in csv_logger
early_stopping =keras.callbacks.EarlyStopping(monitor='val_loss',
                                              min_delta=0, patience=0, verbose=0, mode='auto')
filepath = '4095_0p020_my_model_weights_9layer_0drop.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
model.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_valid, y_valid),callbacks=[checkpoint,csv_logger,tensorboard]) #train

#model.save_weights(filepath)
model.save('4095_0p020_my_model_9layer_0drop.h5')
model.load_weights(filepath)
score = model.evaluate(x_test, y_test, verbose=0) #test
print('Test loss:', score[0])
print('Test accuracy:', score[1])


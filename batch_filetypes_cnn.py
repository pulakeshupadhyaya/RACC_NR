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
import sys
batch_size = 50
num_classes = 4
epochs = 30

import os
#import  os

#import  keras.backend as KTF

import  tensorflow as tf

os.environ["CUDA VISIBLE DEVICES"] = "2,3"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
#config.gpu_options.per_process_gpu_memory_fraction = 0.8
K.set_session(tf.Session(config=config))

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


def load_data(kind,batch_no):
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
    n_batch_size = [0,0,0,0]
    #label_onehot = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    label_counter = [ 1, 1, 1, 1] #read a particular example
    if(kind == "train"):
        num_examples = [0,0,0,0]
        for j in range(4):
            num_examples[j] = batch_size
            n_batch_size[j] = batch_size
        #num_examples = [15087,15089,15088,15027]
        folder_name = "randomized/training/"
    if(kind == "valid"):
        num_examples = [0,0,0,0]
        for j in range(4):
            num_examples[j] = batch_size
            n_batch_size[j] = batch_size
        #num_examples = [20,20,20,20]
        folder_name = "randomized/validation/"
    
    if(kind == "test"):
        #num_examples = [20,20,20,20]
        num_examples = [0,0,0,0]
        for j in range(4):
            num_examples[j] = batch_size
            n_batch_size[j] = batch_size
        folder_name = "randomized/testing/"
    j = 0
    for label in label_list:
        
    
        for i in range(batch_no*n_batch_size[j],(batch_no+1)*n_batch_size[j]):
            #if i % 50 == 0:
            #   print("Loading "+kind+"  example: "+str(i)+" for label : "+label)
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


img_rows, img_cols = 8192, 1 #not one hot
input_shape = (img_rows, img_cols,1)


model = Sequential()
'''model.add(Convolution2D(32, 5,256,
                 activation='relu',
                 input_shape=input_shape))'''
#one hot
model.add(Conv2D(32, (5,1),
              activation='relu',
              input_shape=input_shape))
# 5 x 1 convolution layer
model.add(MaxPooling2D(pool_size=(2, 1), strides=None, padding='valid', data_format=None))
#2 x 1 max pool
model.add(Conv2D(64, (3,1),
                                         activation='relu',
                                         input_shape=input_shape))
#3 x 1 convolution layer
model.add(MaxPooling2D(pool_size=(2, 1), strides=None, padding='valid', data_format=None))
#2 x 1 max pool
model.add(Conv2D(64, (3,1),
                        activation='relu',
                       input_shape=input_shape))
#3 x 1 convolution layer
model.add(MaxPooling2D(pool_size=(2, 1), strides=None, padding='valid', data_format=None))
# 2 x 1 max pool
'''model.add(Convolution2D(64, 3,1,
                        activation='relu',
                        input_shape=input_shape))
#3 x 1 convolution layer
model.add(MaxPooling2D(pool_size=(2, 1), strides=None, padding='valid', data_format=None))
'''
model.add(Flatten())
#flatten
#model.add(Dropout(0.75))
model.add(Dense(num_classes,activation='softmax'))
#dense layer with softmax function
model.compile(loss=keras.losses.categorical_crossentropy,
optimizer=keras.optimizers.Adadelta(),
             metrics=['accuracy'])
#optimization algorithm, metrics, and loss functions
plot_model(model,to_file='batch_model_3.png',show_shapes =  True)


tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
#log data to tensorboard


csv_logger = CSVLogger('batch_results_3.log')
#log accuracy and loss results in csv_logger
early_stopping =keras.callbacks.EarlyStopping(monitor='val_loss',
                                              min_delta=0, patience=0, verbose=0, mode='auto')
last_val_loss = 0

def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

for epoch in range(epochs):
    num_batches = int(15000/batch_size)
    train_acc = 0
    train_loss = 0
    for batch in range(num_batches):
        print("Train Epoch: "+str(epoch)+", example :"+str((batch)*batch_size)+"/"+str(15000),end='\r')
        sys.stdout.flush()
        (x_train, y_train) = load_data("train",batch)
        shuffle_in_unison(x_train,y_train)
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols,1)
        y_train = keras.utils.to_categorical(y_train, num_classes)
        train_score = model.train_on_batch(x_train, y_train)#train
        train_acc = train_score[1]
        train_loss = train_score[0]
    #write_log(csv_logger)

    val_acc = 0.00
    val_loss = 0.00
    num_val_batches = int(2500/batch_size)
    for batch in range(num_val_batches):
        (x_valid, y_valid) = load_data("valid",batch)
        x_valid = x_valid.reshape(x_valid.shape[0], img_rows, img_cols,1)
        y_valid = keras.utils.to_categorical(y_valid, num_classes)
        shuffle_in_unison(x_valid,y_valid)
        val_score = model.test_on_batch(x_valid, y_valid)
        val_acc = val_acc+ val_score[1]
        val_loss = val_loss+ val_score[0]
    val_acc = (val_acc*1.0/num_val_batches)
    val_loss = (val_loss*1.0/num_val_batches)
    print("Epoch= "+str(epoch)+"; train_loss= "+str(train_loss)+"; train_accuracy= "+str(train_acc)+"; Val_loss= "+str(val_loss)+"; val_accuracy= "+str(val_acc))
    sys.stdout.flush()
    if(val_score[0] > last_val_loss and epoch != 0):
        break
    else :
        last_val_loss = val_score[0]


model.save_weights('batch_model_weights_3layer_0drop.h5')
model.save('batch_model_3layer_0drop.h5')
print("Load test data")

test_acc = 0
test_loss = 0
num_test_batches = int(3300/batch_size)
for batch in range(num_test_batches):
    (x_test, y_test) = load_data("test",batch)
    x_test = x_test.reshape(x_test.shape[0],img_rows, img_cols,1)
#print(x_test.shape[0], 'test samples')
    y_test = keras.utils.to_categorical(y_test, num_classes)
#print(y_test.shape[0], 'test labels')
    shuffle_in_unison(x_test,y_test)
    score = model.test_on_batch(x_test, y_test) #test
    test_acc = test_acc+ score[1]
    test_loss = test_loss+ score[0]
test_acc = (test_acc*1.0/num_test_batches)
test_loss = (test_loss*1.0/num_test_batches)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)


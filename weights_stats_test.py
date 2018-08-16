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


model = Sequential()
model = load_model('my_model_3layer_0drop.h5')
model.load_weights('my_model_weights_3layer_0drop.h5')

weights =model.get_weights()[6]
print(len(weights))
sum_weights = [0.0]*64
file = open("weight_results.csv",'w')
file.write("feature_num,html,jpeg,pdf,latex\n")
for i in range(0,len(weights)):
    sum_weights[i%64] = sum_weights[i%64]+weights[i]
for i in range(64):
    sum_weights[i] = sum_weights[i]/1022
for i in range(64):
    file.write(str(i)+",")
    for j in range(4):
        if j < 3:
           file.write(str(sum_weights[i][j])+",")
        else:
           file.write(str(sum_weights[i][j])+"\n")




# coding: utf-8

# In[12]:


import tensorflow as tf
import keras
import numpy as np 
import keras.models
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from math import log
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger


# In[13]:

import os
import tensorflow as tf


os.environ["CUDA VISIBLE DEVICES" ] = "2,3"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6
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
num_test_examples = 4800  #4800
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
        out_folder_name = "4095_noisy_randomized_0p003/training/"
    if(kind == "valid"):
        num_examples = num_valid_examples
        #num_examples = [500,500,500,500]
        folder_name = "4095_randomized/validation/"
        out_folder_name = "4095_noisy_randomized_0p003/validation/"
    if(kind == "test"):
        num_examples = num_test_examples
        #num_examples = [500,500,500,500]
        folder_name = "4095_randomized/testing/"
        out_folder_name = "4095_noisy_randomized_0p003/testing/"
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
        


k = 4095
img_rows, img_cols = k, 1 #not one hot
# the data, shuffled and split between train and test sets
(x_train, y_train) = load_data("train","html")
(x_valid, y_valid) = load_data("valid","html")
(x_test, y_test) = load_data("test","html")

x_train = x_train.reshape(x_train.shape[0], img_rows,img_cols,1)
x_valid = x_valid.reshape(x_valid.shape[0], img_rows,img_cols,1)
x_test = x_test.reshape(x_test.shape[0],img_rows,img_cols,1)
'''
y_train = y_train.reshape(y_train.shape[0], k)
y_valid = y_valid.reshape(y_valid.shape[0], k)
y_test = y_test.reshape(y_test.shape[0],k)
'''
print('After reshape:')
print('x_train shape:', x_train.shape)
print('x_valid shape:', x_valid.shape)
print('x_test shape:', x_test.shape)

input_shape = (img_rows, img_cols,1)



# convert class vectors to binary class matrices

print('Shuffling in unison ')
shuffle_in_unison(x_train,y_train)
shuffle_in_unison(x_valid,y_valid)
shuffle_in_unison(x_test,y_test)
#print('y_train shape:', y_train.shape)
#print(y_train.shape[0], 'train labels')
#print(y_test.shape[0], 'test labels')



# In[14]:


def negGrowthRateLoss(b,q):
    return (K.mean(-K.log(b +pow(-1,b)*q)/K.log(2.0)))


# In[15]:


batch_size = 25
epochs = 8


#model.add(Dense(1,activation='sigmoid'))
model = Sequential()
#model.add(Dense(k,activation='sigmoid',input_dim=k))
#model.add(Dropout(0.75))
model.add(Conv2D(32, (3,1),
              activation='relu',
              input_shape=input_shape))
# 3 x 1 convolution layer

model.add(MaxPooling2D(pool_size=(2, 1), strides=None, padding='valid', data_format=None))
'''
model.add(Conv2D(8, (5,1),
              activation='relu',
              input_shape=input_shape))
# 3 x 1 convolution layer
model.add(MaxPooling2D(pool_size=(2, 1), strides=None, padding='valid', data_format=None))

model.add(Conv2D(16, (5,1),
              activation='relu',
              input_shape=input_shape))
# 3 x 1 convolution layer
model.add(MaxPooling2D(pool_size=(2, 1), strides=None, padding='valid', data_format=None))
model.add(Conv2D(32, (5,1),
              activation='relu',
              input_shape=input_shape))
# 3 x 1 convolution layer
model.add(MaxPooling2D(pool_size=(2, 1), strides=None, padding='valid', data_format=None))

model.add(Conv2D(64, (5,1),
              activation='relu',
              input_shape=input_shape))
# 3 x 1 convolution layer
model.add(MaxPooling2D(pool_size=(2, 1), strides=None, padding='valid', data_format=None))
'''
model.add(Flatten())
model.add(Dense(k,activation='sigmoid'))
model.compile(loss=negGrowthRateLoss,optimizer=keras.optimizers.Adadelta())
#optimization algorithm, metrics, and loss functions
filepath = 'test_weights_1_0_3w.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
csv_logger = CSVLogger('losses_1_0_3w.csv')
plot_model(model,to_file='model_1_0_3w'+str(k)+'.png',show_shapes =  True)

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_valid, y_valid),callbacks=[checkpoint,csv_logger]) #train

#model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1)


predictions = model.predict(x_test,verbose=0)
model.save('model_1_0_3w_save.h5')
model.load_weights(filepath)
p_p_y = np.array([[0.0,0.0],[0.0,0.0]])
ct = np.array([0.0,0.0])
p_p = np.array([0.0,0.0])
p_y = np.array([0.0,0.0])
for j in range(0,num_test_examples):
    #filename = open("results/res_test_exmpl_"+str(j)+".csv",'w')
    for i in range(0,4095):
        
        #filename.write(str(predictions[j][i])+","+str(y_test[j][i])+","+str(x_test[j][i])+"\n")
        #print(predictions[j][i])
        if( y_test[j][i] == 0):
            p_p_y[0][0] = p_p_y[0][0]+predictions[j][i]
	    #p[1][y_test[j][i]] = p[1][y_test[j][i]]+(1.0-predictions[j][i])
            ct[0] = ct[0]+1.0
        else:
            p_p_y[0][1] = p_p_y[0][1]+predictions[j][i]
	    #p[0][y_test[j][i]] = p[0][y_test[j][i]]+(1.0-predictions[j][i])
            ct[1] = ct[1]+1.0
        p_p[0] = p_p[0]+predictions[j][i]

    #filename.close()

p_p_y[0][0] = p_p_y[0][0]/ct[0]
p_p_y[0][1] = p_p_y[0][1]/ct[1]

p_p_y[1][0] = 1.0 - p_p_y[0][0]
p_p_y[1][1] = 1.0 - p_p_y[0][1]


p_p[0] = p_p[0]/(ct[0]+ct[1])
p_p[1] = 1-p_p[0]

p_y[0] = ct[0]/(ct[0]+ct[1])
p_y[1] = ct[1]/(ct[0]+ct[1])

mut_inf = 0.0

for i in range(0,2):
    for j in range(0,2):
        p_p_y[i][j] = p_p_y[i][j]*p_y[j]
        

#ct = ct/sum(ct)
#ct_pred = ct_pred/sum(ct_pred)

file = open("results_1_0_3w.txt",'w+')
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







import numpy as np
from random import randint
import random
import time

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

def send_channel(bitfield_arr):
    newarr = [0,0,0,0,0,0,0,0]
    for i in range(len(bitfield_arr)):
        r = random.uniform(0, 1)
        if(r < 0.008):
            
            if(bitfield_arr[i] == 1):
                newarr[i] = 0
            if(bitfield_arr[i] == 0):
                newarr[i] = 1
    
        else:
            newarr[i] = bitfield_arr[i]
    sum = 0
    n = len(newarr)
    for i in range(n):
        sum = sum+pow(2,n-i-1)*newarr[i]
    return(sum)
def bitfield(n):
    integers = [0,0,0,0,0,0,0,0]
    
    listed =  [int(digit) for digit in bin(n)[2:]]
    #print(listed)
    count = 7
    for i in range(len(listed)-1,-1,-1):
        integers[count] = listed[i]
        count = count-1
    return(integers)
def load_data(kind):
    #this function loads data from the
    #files & returns numpy arrays of test and training examples
    '''ly = []
        for i in range(0,256):
        ly.append(i)
        
        a = np.array(ly)
        
        b = np.zeros((256, 256))
        b[np.arange(256), a] = 1'''
    
    
    
    
    label_list = ['html','jpeg','pdf','latex'] #list of labels
    #label_onehot = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    label_counter = [ 1, 1, 1, 1] #read a particular example
    if(kind == "train"):
        #num_examples = [100,100,100,100]
        num_examples = [15087,15089,15088,15027]
        folder_name = "randomized/training/"
        outfolder_name = "noisy_randomized_0p008/training/"
    if(kind == "valid"):
        num_examples = [2514,2525,2514,2504]
        #num_examples = [20,20,20,20]
        folder_name = "randomized/validation/"
        outfolder_name = "noisy_randomized_0p008/validation/"
    if(kind == "test"):
        #num_examples = [20,20,20,20]
        num_examples = [3398,3385,3397,3458]
        folder_name = "randomized/testing/"
        outfolder_name = "noisy_randomized_0p008/testing/"
    j = 0
    for label in label_list:
        print(j)
        now = time.time()
        for i in range(num_examples[j]):
	   
            if i % 100 == 0:
		later = time.time()
            	print(later-now)
                print("Loading "+kind+"  example: "+str(i)+" for label : "+label)

            f_out = open(outfolder_name+label+"/"+str(i)+".txt", 'wb+')
            with open(folder_name+label+"/"+str(i)+".txt", 'rb') as f:
                count  = 0
                #l = []
                for c in get_next_character(f) :
                    
                    #l.append(b[ord(c)]) #one hot
                    bitfield_arr = bitfield(ord(c))
                    
                    sum = send_channel(bitfield_arr)
                    slist = []
                    slist.append(sum)
		    #print(chr(slist[0]))
                    f_out.write(chr(slist[0]))
                    #l.append(ord(c)*1.0/255.0)
                    count = count+1

            f_out.close()
        j = j+1

#read shuffled data and split between train and test sets


#
load_data("train")
load_data("valid")
load_data("test")


import os

for i in range(0,1000):
        os.system('CUDA_VISIBLE_DEVICES=5 python DNN_end_to_end_0p01.py '+str(i)+' '+'html')



for i in range(0,1000):
	os.system('CUDA_VISIBLE_DEVICES=5 python DNN_end_to_end_0p01.py '+str(i)+' '+'jpeg')



for i in range(0,1000):
        os.system('CUDA_VISIBLE_DEVICES=5 python DNN_end_to_end_0p01.py '+str(i)+' '+'pdf')

for i in range(0,1000):
        os.system('CUDA_VISIBLE_DEVICES=5 python DNN_end_to_end_0p01.py '+str(i)+' '+'latex')



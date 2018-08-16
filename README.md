# training code
4095_filetypes_cnn.py is the program for training a file type recognition system of 4095 bit segments of files.

DNN_conv_deconv.py is used to train a convolution-deconvolution network to provide soft information using 
natural redundancy(NR) which can be supplemented with LPDC decoding for error correction.

# testing code
DNN_end_to_end_test.py tests an end to end system which does file type recognition based on the system trained by 
4095_filetypes_cnn.py,and based on the result, uses the corresponding convolution-deconvolution network trained by 
DNN_conv_deconv.py. It then stores the soft information so that it is available for the new LDPC decoder

LDPC_ED_BSC.cpp tests the new LDPC decoding aided by NR-based soft information stored by DNN_end_to_end_test.py

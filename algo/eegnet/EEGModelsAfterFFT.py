"""
__file__ :  'EEGModelsAfterFFT.py'
Adapted from EEGModels.py 
Removed the first Conv2D (F1, (1, kernLength layer)), as FFT is to be performed before feeding to model.
Input is to be with shape (None, 1, channel_num, bin_num), instead of previously (None, 1, channel_num, sample_num)
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, MaxPool2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
import tensorflow as tf
import pdb
import numpy as np

# GTL: dropoutRate changed from 0.5 to 0.25
def EEGNet(nb_classes, Chans = 64, Samples = 128, 
             dropoutRate = 0.5, kernLength = 16, F1 = 64, dense0=64,
             D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout', 
             trials=128, regularize = [False, 'L1', 0.01]):
    """ Keras Implementation of EEGNet
    Inputs:
      nb_classes      : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer. We found
                        that setting this to be half the sampling rate worked
                        well in practice. For the SMR dataset in particular
                        since the data was high-passed at 4Hz we used a kernel
                        length of 32.     
      F2              : number of pointwise filters (F2) to learn. 
      D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
      dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.

    """
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1   = Input(shape = (1, Chans, Samples))
    print('input1 shape = {}. Chans = {}, Samples = {}'.format(input1.shape, Chans, Samples))
    
#     block1       = Conv2D(32, (1, 4), padding = 'same', strides=(1,2),
#                                        input_shape = (1, Chans, Samples),
#                                        use_bias = False,
#                                        data_format='channels_first')(input1)   # By GTL channel_first
#     block1       = Conv2D(64, (1, 4), padding = 'same', strides=(1,2),
#                                        input_shape = (1, Chans, Samples),
#                                        use_bias = False,
#                                        data_format='channels_first')(block1)   # By GTL channel_first
        
    # Changed such that take input1 instead of block1
    if regularize[0] == False:
#         block1       = DepthwiseConv2D((Chans, 1), use_bias = False, 
#                                        depth_multiplier = D, data_format='channels_first', # By GTL channel_first
#                                        depthwise_constraint = max_norm(1.))(block1)
        block1       = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                       depth_multiplier = D, data_format='channels_first', # By GTL channel_first
                                       depthwise_constraint = max_norm(1.))(input1)
        print('block1 second stage shape = {}. Chans = {}, D = {}'.format(block1.shape, Chans, D))
    elif regularize[1] == 'L1': 
        block1       = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                       depth_multiplier = D, data_format='channels_first', # By GTL channel_first
                                       depthwise_regularizer = tf.keras.regularizers.L1(regularize[2]), 
                                       depthwise_constraint = max_norm(1.))(input1)
    elif regularize[1] == 'L2':
        block1       = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                       depth_multiplier = D, data_format='channels_first', # By GTL channel_first
                                       depthwise_regularizer = tf.keras.regularizers.L2(regularize[2]), 
                                       depthwise_constraint = max_norm(1.))(input1)
    block1       = BatchNormalization(axis = 1)(block1)
    block1       = Activation('elu')(block1)
#     block1       = AveragePooling2D((1, 4),data_format='channels_first')(block1) # Original EEGNet
#     block1       = MaxPool2D((1, 4),data_format='channels_first')(block1)  # Changed
    print('block1 avg pool stage shape = {}. AveragePooling2D Params (hardcoded)= {}'.format(block1.shape, (1,4)))
    block1       = dropoutType(dropoutRate)(block1)  
    print('block1 dropout stage shape = {}. dropoutRate = {}'.format(block1.shape, dropoutRate))
    
    if regularize[0] == False:
        # GTL: (F2, (1, 16)) changed to (F2, (1, 512))
        block2       = SeparableConv2D(F2, (1, 4),  
                                       use_bias = False, padding = 'same',data_format='channels_first')(block1) 
        print('block2 1st stage shape = {}. F2 = {}, param_2 (hardcoded)= {}'.format(block2.shape, F2, (1,16)))
    elif regularize[1] == 'L1':
        block2       = SeparableConv2D(F2, (1, 4),  
                                       use_bias = False, 
                                       pointwise_regularizer = tf.keras.regularizers.L1(regularize[2]),
                                       padding = 'same',data_format='channels_first')(block1) 
    elif regularize[1] == 'L2':
        block2       = SeparableConv2D(F2, (1, 4),  
                                       use_bias = False, 
                                       pointwise_regularizer = tf.keras.regularizers.L2(regularize[2]),
                                       padding = 'same',data_format='channels_first')(block1) 
     
    block2       = BatchNormalization(axis = 1)(block2)
    block2       = Activation('elu')(block2)
#     block2       = AveragePooling2D((1, 8),data_format='channels_first')(block2) # Original EEGNet
    print('block2 avg pool stage shape = {}. AveragePooling2D params (hardcoded) = {}'.format(block2.shape, (1,8)))
#     block2       = MaxPool2D((1, 8),data_format='channels_first')(block2) # Changed
    block2       = dropoutType(dropoutRate)(block2)
    print('block2 dropout stage shape = {}. dropoutRate = {}'.format(block2.shape, dropoutRate))
        
    flatten      = Flatten(name = 'flatten')(block2)
    print('flatten shape = {}'.format(flatten.shape))
    
    # Added additional dense layer
    dense = Dense(dense0, name = 'dense_0')(flatten)
    dense = Dense(32, name = 'dense_1')(dense)
#     dense = Dense(dense0, name = 'dense_0', 
#                              kernel_constraint = max_norm(norm_rate))(flatten)
#     dense = Dense(64, name = 'dense_1', 
#                              kernel_constraint = max_norm(norm_rate))(dense)
    
    if regularize[0] == False:    
        dense        = Dense(nb_classes, name = 'dense')(dense)
#         dense        = Dense(nb_classes, name = 'dense', 
#                              kernel_constraint = max_norm(norm_rate))(dense)
        print('dense shape = {}. nb_classes = {}, norm_rate = {}'.format(dense.shape, nb_classes, norm_rate))
    elif regularize[1] == 'L1':
        dense        = Dense(nb_classes, name = 'dense', 
                             kernel_regularizer = tf.keras.regularizers.L1(regularize[2]),
                             kernel_constraint = max_norm(norm_rate))(dense)
    elif regularize[1] == 'L2':
        dense        = Dense(nb_classes, name = 'dense', 
                             kernel_regularizer = tf.keras.regularizers.L2(regularize[2]),
                             kernel_constraint = max_norm(norm_rate))(dense)
    
    softmax      = Activation('softmax', name = 'softmax')(dense)
    ## Original ##################################################################
    
    ## Changed to follow SCU #####################################################
#     F1 = 16
#     kernLength = 5
#     dropoutRate = 0.5
    
#     input1   = Input(shape = (1, Chans, Samples))

    
#     block1       = Conv2D(F1, (1, kernLength), padding = 'same',
#                                    input_shape = (1, Chans, Samples),
#                                    use_bias = False, strides = (1, 2),
#                                    data_format='channels_first')(input1)   # By GTL channel_first
#     block1       = BatchNormalization(axis = 1)(block1)  
#     block1       = Activation('relu')(block1)
#     block1       = MaxPool2D((1, 2),data_format='channels_first')(block1)
#     block1       = dropoutType(dropoutRate)(block1)  
    
    
#     block1       = Conv2D(32, (1, kernLength), padding = 'same',
#                                    use_bias = False, strides = (1, 2),
#                                    data_format='channels_first')(block1)  
#     block1       = BatchNormalization(axis = 1)(block1)  
#     block1       = Activation('relu')(block1)
#     block1       = MaxPool2D((1, 2),data_format='channels_first')(block1)
#     block1       = dropoutType(dropoutRate)(block1)  
    
    
#     block1       = Conv2D(64, (1, kernLength), padding = 'same',
#                                    use_bias = False, strides = (1, 2),
#                                    data_format='channels_first')(block1)  
#     block1       = BatchNormalization(axis = 1)(block1)  
#     block1       = Activation('relu')(block1)
#     block1       = MaxPool2D((1, 2),data_format='channels_first')(block1)
#     block1       = dropoutType(dropoutRate)(block1)  
    
    
#     block1       = Conv2D(128, (1, kernLength), padding = 'same',
#                                    use_bias = False, strides = (1, 2),
#                                    data_format='channels_first')(block1)  
#     block1       = BatchNormalization(axis = 1)(block1)  
#     block1       = Activation('relu')(block1)
#     block1       = MaxPool2D((1, 2),data_format='channels_first')(block1)
#     block1       = dropoutType(dropoutRate)(block1)  
    
      
        
#     flatten      = Flatten(name = 'flatten')(block1)
    
#     dense        = Dense(600, name = 'dense1', 
#                          kernel_constraint = max_norm(norm_rate))(flatten)
#     dense       = Activation('relu')(dense)
#     dense       = dropoutType(dropoutRate)(dense)  
    
#     dense        = Dense(60, name = 'dense2', 
#                          kernel_constraint = max_norm(norm_rate))(dense)
#     dense       = Activation('relu')(dense)
#     dense       = dropoutType(dropoutRate)(dense)  
        
#     dense        = Dense(nb_classes, name = 'dense', 
#                          kernel_constraint = max_norm(norm_rate))(dense)

    
#     softmax      = Activation('softmax', name = 'softmax')(dense)
    ## Changed to follow SCU #####################################################
    
    
    return Model(inputs=input1, outputs=softmax)

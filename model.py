import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Lambda, Dense, Flatten,MaxPooling2D, ZeroPadding2D,  Dropout, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.losses import binary_crossentropy
import numpy.random as rng
import numpy as np
import os

def initialize_weights(shape, dtype=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer weights with mean as 0.0 and standard deviation of 0.01
    """
    return np.random.normal(loc = 0.0, scale = 1e-2, size = shape)

def initialize_bias(shape, dtype=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer bias with mean as 0.5 and standard deviation of 0.01
    """
    return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)


def Siamese(input_shape):
    model = Sequential()
    
    
    model.add(Conv2D(96, kernel_size=(9,9), activation='relu', name='conv1_1', strides=1, input_shape=input_shape,
                     kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4)))
    model.add(BatchNormalization(epsilon=1e-06, axis=1, momentum=0.9))
    model.add(MaxPooling2D((3,3),(2,2)))
    
    
    

    model.add(Conv2D(256, kernel_size=(5,5), activation='relu', name='conv2_1', strides=1, input_shape=input_shape,
                     kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4)))
    model.add(BatchNormalization(epsilon=1e-06, axis=1, momentum=0.9))
    model.add(MaxPooling2D((3,3),(2,2)))
    
    
    
    
    model.add(Conv2D(384, kernel_size=(3,3), activation='relu', name='conv3_1', strides=1, input_shape=input_shape,
                 kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4)))
    model.add(Conv2D(384, kernel_size=(3,3), activation='relu', name='conv3_2', strides=1, input_shape=input_shape,
                 kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4)))
    model.add(Conv2D(256, kernel_size=(3,3), activation='relu', name='conv3_3', strides=1, input_shape=input_shape,
                 kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D((3,3), (2,2)))
    model.add(Dropout(0.3))
    
    
    model.add(Flatten(name='Flatten'))
#    model.add(Dense(1024, kernel_regularizer=l2(0.0005), activation='relu', kernel_initializer=initialize_weights))

#    model.add(Dense(128, kernel_regularizer=l2(0.0005), activation='relu', kernel_initializer=initialize_weights))
    model.add(Dense(2304, kernel_regularizer=l2(0.0005), activation='relu', kernel_initializer=initialize_weights))
    model.add(Dropout(0.5))
    model.add(Dense(240, kernel_regularizer=l2(0.0005), activation='relu', kernel_initializer=initialize_weights))

    return model

# -*- coding: utf-8 -*-

from keras.layers import Input, Conv2D,Conv1D, Lambda, Dense, Flatten,MaxPooling2D,MaxPooling1D,Dropout, BatchNormalization
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.regularizers import l2
from keras.losses import binary_crossentropy
import numpy.random as rng
from itertools import combinations, product
import numpy as np
import os
import time
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


def euclidean_distance(vects):
    """Find the Euclidean distance between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=-1, keepdims=True)
    # return K.sqrt(K.maximum(sum_square, K.epsilon()))
    return K.sqrt(sum_square)


def cosine_similarity(vectors):
    x, y = vectors
    dot_product = K.sum(x * y, axis=-1, keepdims=True)
    magnitude_x = K.sqrt(K.sum(K.square(x), axis=-1, keepdims=True))
    magnitude_y = K.sqrt(K.sum(K.square(y), axis=-1, keepdims=True))
    return dot_product / (magnitude_x * magnitude_y + K.epsilon())


def contrastive_loss(y_true, y_pred):
    margin = 1
    # Clip y_pred to prevent numerical instability
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * margin_square)


def accuracy_metric(y_true, y_pred):
    threshold = 0.5
    y_pred_binary = K.cast(K.less(y_pred, threshold), K.floatx())  # Convert distance to binary
    return K.mean(K.equal(y_true, y_pred_binary))

def load_siamese_net_2D(input_shape = (75,75,3)):
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    # WDCNN
    convnet = Sequential()

    convnet.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='valid',input_shape=input_shape))
    convnet.add(MaxPooling2D((2,2)))

    convnet.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='valid'))
    convnet.add(MaxPooling2D((2,2)))

    convnet.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='valid'))
    convnet.add(MaxPooling2D((2,2)))

    convnet.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
    convnet.add(MaxPooling2D((2,2)))

    convnet.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
    convnet.add(MaxPooling2D((2,2)))

    # convnet.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
    # convnet.add(MaxPooling2D((2,2)))

    convnet.add(Flatten())
    convnet.add(Dense(100,activation='sigmoid'))

    #call the convnet Sequential model on each of the input tensors so params will be shared
    encoded_l = convnet(left_input)
    encoded_r = convnet(right_input)
    #layer to merge two encoded inputs with the l1 distance between them
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    #call this layer on list of two input tensors.
    L1_distance = L1_layer([encoded_l, encoded_r])
    D1_layer = Dropout(0.5)(L1_distance)
    prediction = Dense(1,activation='sigmoid')(D1_layer)
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

    optimizer = Adam()

    siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer)

    # print('\nsiamese_net summary:')
    # siamese_net.summary()
    
    return siamese_net

def load_siamese_net_1D(input_shape = (2048,2)):
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    convnet = Sequential()

    # WDCNN
    # convnet.add(BatchNormalization(input_shape=input_shape))
    convnet.add(Conv1D(filters=16, kernel_size=64, strides=16, activation='relu', padding='same',input_shape=input_shape))
    convnet.add(MaxPooling1D(strides=2))
    
    convnet.add(Conv1D(filters=32, kernel_size=3, strides=1, activation='relu', padding='same'))
    convnet.add(MaxPooling1D(strides=2,padding='same'))
    
    convnet.add(Conv1D(filters=64, kernel_size=2, strides=1, activation='relu', padding='same'))
    convnet.add(MaxPooling1D(strides=2,padding='same'))
    
    convnet.add(Conv1D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same'))
    convnet.add(MaxPooling1D(strides=2,padding='same'))
    
    convnet.add(Conv1D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same'))
    convnet.add(MaxPooling1D(strides=2, padding='same'))
    
    convnet.add(Flatten())
    # convnet.add(BatchNormalization())
    # convnet.add(Dense(100,activation='sigmoid'))
    convnet.add(Dense(100, activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=l2(0.01)))



    print('WDCNN convnet summary:')
    convnet.summary()

    #call the convnet Sequential model on each of the input tensors so params will be shared
    encoded_l = convnet(left_input)
    encoded_r = convnet(right_input)
    
    ##layer to merge two encoded inputs with the l1 distance between them
    
    ## Simple abs difference
    # L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    
    ## euclidean_distance
    L1_layer = Lambda(euclidean_distance)
    
    ## Cosine_disance
    # L1_layer = Lambda(cosine_similarity,output_shape=(1,))
    
    #call this layer on list of two input tensors.
    L1_distance = L1_layer([encoded_l, encoded_r])
    D1_layer = Dropout(0.5)(L1_distance)
    prediction = Dense(1,activation='sigmoid')(D1_layer)
    siamese_net = Model(inputs=[left_input,right_input],outputs=D1_layer)

    # optimizer = Adam(0.0001)
    optimizer = Adam(0.0001)
    #//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
    
    ## binary loss
    # siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer)
    
    ## contrastive_loss
    siamese_net.compile(loss=contrastive_loss, optimizer=optimizer, metrics=[accuracy_metric])
    
    
    
#     print('\nsiamese_net summary:')
#     siamese_net.summary()
#     print(siamese_net.count_params())
    
    return siamese_net


def load_wdcnn_net_1D(input_shape=(200, 3), nclasses=5):
    left_input = Input(input_shape)
    convnet = Sequential()

    # WDCNN
    convnet.add(Conv1D(filters=16, kernel_size=8, strides=2, activation='relu', 
                       padding='same', kernel_initializer='he_normal', input_shape=input_shape))
    convnet.add(BatchNormalization())
    convnet.add(MaxPooling1D(pool_size=2, strides=2))

    convnet.add(Conv1D(filters=32, kernel_size=4, strides=1, activation='relu', padding='same', kernel_initializer='he_normal'))
    convnet.add(BatchNormalization())
    convnet.add(MaxPooling1D(pool_size=2, strides=2))

    convnet.add(Conv1D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same', kernel_initializer='he_normal'))
    convnet.add(BatchNormalization())
    convnet.add(MaxPooling1D(pool_size=2, strides=2))

    convnet.add(Conv1D(filters=64, kernel_size=3, strides=1, activation='relu', kernel_initializer='he_normal'))
    convnet.add(BatchNormalization())
    convnet.add(MaxPooling1D(pool_size=2, strides=2))

    convnet.add(Flatten())
    convnet.add(Dense(100, activation='relu', kernel_initializer='he_normal'))

    # Connect the functional input
    x = convnet(left_input)  # Pass the functional input through the Sequential model
    x = Dropout(0.5)(x)      # Add Dropout layer
    prediction_cnn = Dense(nclasses, activation='softmax')(x)  # Output layer

    # Create functional model
    wdcnn_net = Model(inputs=left_input, outputs=prediction_cnn)

    # Compile the model
    optimizer = Adam()
    wdcnn_net.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return wdcnn_net



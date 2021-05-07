##========== Copyright (c) 2020-2021, Adam Lanicek & ANALYTICSINDIAMAG.COM, All rights reserved. =========##
##
## Purpose:     Module for evaluating the already trained AlexNet CNN with
##              weights stored in cnnInputs/ directory. 
## 
##              When run with --fakeConv parameter the standard convolutional layers 
##              are substituted with the FakeApproxConv2D layers from the 
##              fake_approx_convolutional.py module from the tf-approximate library
##              (https://github.com/ehw-fit/tf-approximate. 
##
## Implement.:  The AlexNet model structure was taken from https://laptrinhx.com/hands-on-
##              guide-to-implementing-alexnet-with-keras-for-multi-class-image-
##              classification-3201492782/ and modified with the FakeApproxConv2D layers
##              by Adam Lanicek based on inspiration from the examples
##              stored in the tf-approximate library.
##
## $Date:       $2021-05-05
##=========================================================================================================##

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import tensorflow as tf
import argparse
from keras.utils import to_categorical

from fake_approx_convolutional import FakeApproxConv2D

# cuDNN can sometimes fail to initialize when TF reserves all of the GPU memory
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[0:1], 'GPU')

# Process arguments
parser = argparse.ArgumentParser()
parser.add_argument('--fakeConv', action='store_true')
parser.add_argument('--m1', type=str, help='Approximate multiplication table for layer 1 (8x8)', default='')
parser.add_argument('--m2', type=str, help='Approximate multiplication table for layer 2 (8x8)', default='')
parser.add_argument('--m3', type=str, help='Approximate multiplication table for layer 3 (8x8)', default='')
parser.add_argument('--m4', type=str, help='Approximate multiplication table for layer 4 (8x8)', default='')
parser.add_argument('--m5', type=str, help='Approximate multiplication table for layer 5 (8x8)', default='')

args = parser.parse_args()

#Instantiation
AlexNet = Sequential()

#1st (Fake-)Convolutional Layer
if args.fakeConv:
    AlexNet.add(FakeApproxConv2D(filters=96, input_shape=(32,32,3), kernel_size=(11,11), strides=(4,4), padding='same', mul_map_file=args.m1))
else:
    AlexNet.add(Conv2D(filters=96, input_shape=(32,32,3), kernel_size=(11,11), strides=(4,4), padding='same'))

AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))
AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

#2nd (Fake-)Convolutional Layer
if args.fakeConv:
    AlexNet.add(FakeApproxConv2D(filters=256, kernel_size=(5, 5), strides=(1,1), padding='same', mul_map_file=args.m2))
else:
    AlexNet.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1,1), padding='same'))

AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))
AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

#3rd (Fake-)Convolutional Layer
if args.fakeConv:
    AlexNet.add(FakeApproxConv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same', mul_map_file=args.m3))
else:
    AlexNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))

AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))

#4th (Fake-)Convolutional Layer
if args.fakeConv:
    AlexNet.add(FakeApproxConv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same', mul_map_file=args.m4))
else:
    AlexNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))

AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))

#5th (Fake-)Convolutional Layer
if args.fakeConv:
    AlexNet.add(FakeApproxConv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same' , mul_map_file=args.m5))
else:
    AlexNet.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))

AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))
AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

#Passing it to a Fully Connected layer
AlexNet.add(Flatten())
# 1st Fully Connected Layer
AlexNet.add(Dense(4096, input_shape=(32,32,3,)))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))
# Add Dropout to prevent overfitting
AlexNet.add(Dropout(0.4))

#2nd Fully Connected Layer
AlexNet.add(Dense(4096))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))
#Add Dropout
AlexNet.add(Dropout(0.4))

#3rd Fully Connected Layer
AlexNet.add(Dense(1000))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))

#Add Dropout
AlexNet.add(Dropout(0.4))

#Output Layer
AlexNet.add(Dense(10))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('softmax'))

#Model Summary
#AlexNet.summary()

# Compiling the model
AlexNet.compile(loss = keras.losses.categorical_crossentropy, optimizer= 'adam', metrics=['accuracy'])
AlexNet.load_weights('cnnInputs/AlexNet_weights_norm')

print('================================================================================')
if args.fakeConv:
    print('Testing approximate model with\nConvLayer 1: {}\nConvLayer 2: {}\nConvLayer 3: {}\nConvLayer 4: {}\nConvLayer 5: {}'
            .format(args.m1, args.m2, args.m3, args.m4, args.m5))
else:
    print('Testing standard model')

data = np.load('cnnInputs/test_data.npz')

loss, acc = AlexNet.evaluate(data['x_test'][:500], data['y_test'][:500], verbose=1)
print('Test loss: {:.6f}'.format(loss))
print('Test accuracy: {:.6f}'.format(acc))
##========== Copyright (c) 2021, Adam Lanicek, All rights reserved. =========##
##
## Purpose:     Download and prepare the data for the training & evaluation of
##              the CNN with approximate layers
##
## $Date:       $2021-05-05
##============================================================================##

from keras.datasets import cifar10
import numpy as np
import tensorflow as tf
import shutil
import os
from sklearn.utils.multiclass import unique_labels
from keras.utils import to_categorical

# Path to the stored and preprocessed numpy arrays
CNN_INP_DIR = 'cnnInputs'
if not os.path.isdir("cnnInputs"):
    os.makedirs("cnnInputs")

DATA_PATH = os.path.join(CNN_INP_DIR, 'test_data.npz')
(x_train, y_train),(x_test, y_test)=cifar10.load_data()

x_test = x_test.reshape(10000, 32, 32, 3).astype('float32') / 255
y_test = y_test.astype('float32')
y_test = to_categorical(y_test)

np.savez(DATA_PATH, x_test=x_test, y_test=y_test)

data = np.load(DATA_PATH)
print('Downloaded & normalized data:')
print((x_test.shape,y_test.shape))
print('Data stored & loaded again from disk')
print((data['x_test'].shape,data['y_test'].shape))


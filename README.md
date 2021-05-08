# Approximation of neural networks

## BIN project at BUT FIT 2020/2021

Files completely authored by Adam Lanicek:
- **testSetLoad.py**
- **nsga2.py**
- **optimSetup.py**
- **Makefile**

Files partially authored by Adam Lanicek:
- **train_AlexNet.py**
- **eval_AlexNet.py**

Files taken from other sources:
- **fake_approx_convolutional.py**

### Files overview

- **testSetLoad.py** - loads and preprocesses the CIFAR-10 dataset for training and evaluation

- **train_AlexNet.py --weights outputWeightFile** - trains the AlexNet CNN 

- **eval_AlexNet.py --weights outputWeightFile [--fakeConv] [-m1] [multiplierBinFile] [-m2] [multiplierBinFile] [-m3] [multiplierBinFile] [-m4] [multiplierBinFile] [-m5] [multiplierBinFile]** - runs the inference on the test set either with the standard or approximate layers (if --fakeConv launch argument is provided)

- **fake_approx_convolutional.py** - copied from https://github.com/ehw-fit/tf-approximate/tree/master/tf2/python/keras/layers

- **nsga2.py** - contains implementation of the NSGA-II algorithm using the pymoo Python library

- **optimSetup.py** - contains the necessary information regarding the amount of multiplications in AlexNet convolutional layers as well as energy requirements of the multipliers


## Make commands description

- **make approxLib** - downloads the prebuilt binary libApproxGPUOpsTF.so (see also https://github.com/ehw-fit/tf-approximate/tree/master/tf2) for CUDA architecture 6.1 **(will not work on other architectures!)**

- **make multipliersReady**: clones the tf-approximate git repo to have access to the multipliers


Before running the following commands, please **make sure you have prepared the runtime environment
(tested using Tensorflow 2.2.0 na CUDA 10.1)**

- **make train** - runs a very short training of the AlexNet training for the demonstration purposes

- **make std_eval** - runs a test inference using the standard convolutional layers

- **make approx_eval** - runs a test inference using the **approximate** layers substituting the convolutional ones

- **make nsga2** - runs the NSGA-II algorithm to find the optimal set of the 




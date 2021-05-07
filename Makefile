##========== Copyright (c) 2021, Adam Lanicek, All rights reserved. =========##
##
## Purpose:     Makefile to prepare the environment, train the CNN and evaluate it
##
## Implement.:  Prepared by Adam Lanicek.
##
## $Date:       $2021-05-07
##============================================================================##

#### GENERAL SETTINGS - DO NOT EDIT
CONDA_REQS=anaconda_requirements.txt
PIP_REQS=python_requirements.txt
MULTS_PATH=tf-approximate/tf2/examples/axmul_8x8/mul8u_
MULTS_SUFF=.bin

M1=${MULTS_PATH}_${MULT1_NAME}${MULTS_SUFF}
M2=${MULTS_PATH}_${MULT2_NAME}${MULTS_SUFF}
M3=${MULTS_PATH}_${MULT3_NAME}${MULTS_SUFF}
M4=${MULTS_PATH}_${MULT4_NAME}${MULTS_SUFF}
M5=${MULTS_PATH}_${MULT5_NAME}${MULTS_SUFF}
#### -----------------------------------------------------------------------------

#### MODIFIABLE SETTINGS
WEIGHTS_NAME=myAlexNetWeights
TRAIN_EPOCHS=3
CONDA_ENV_NAME=binProjEnv

# Test multipliers ONLY for the purposes of the approx_eval target - can be edited
MULT1_NAME=EXZ
MULT2_NAME=125K
MULT3_NAME=2P7
MULT4_NAME=KEM
MULT5_NAME=ZFB
#### -----------------------------------------------------------------------------

# NSGA-II ALGORITHM SETTINGS
POPULATION_SIZE = 2
INDIVIDUAL_CNT = 5


all:
	conda create --name ${CONDA_ENV_NAME} --file ${CONDA_REQS}
	pip install -r ${PIP_REQS}
	mkdir cnnInputs
	wget --no-check-certificate -O './cnnInputs/libApproxGPUOpsTF.so' 'https://drive.google.com/uc?export=download&id=1FLB5HJ-qGM-akTcRWBglXN4pUokOWJXr'
	@echo "Make sure the compiled libApproxGPUOpsTF.so library is placed in the cnnInputs directory!"

train: all
	@python testSetLoad.py
	@echo "Training AlexNet for 3 epochs......."
	@python train_AlexNet.py --WEIGHTS_NAME ${WEIGHTS_NAME} --epochs ${TRAIN_EPOCHS}

multipliersReady:
	@echo "Cloning tf-approximate repository to have access to the multipliers......."
	@git clone https://github.com/ehw-fit/tf-approximate.git

std_eval: all train
	@echo "Evaluating AlexNet with standard convolution layers......."
	python eval_AlexNet.py --weights ${WEIGHTS_NAME}

approx_eval: all train multipliersReady 
	python eval_AlexNet.py --weights ${WEIGHTS_NAME} --fakeConv --m1 ${M1} --m2 ${M2} --m3 ${M3} --m4 ${M4} --m5 ${M5}

nsga2: train multipliersReady
	python nsga2.py --weights ${WEIGHTS_NAME} --pop_size ${POPULATION_SIZE} --ind_cnt ${INDIVIDUAL_CNT}

pack:
	zip -r xlanic04.zip Makefile *.py *.txt runsOutput README.md requirements.txt
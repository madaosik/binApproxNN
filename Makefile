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
MULTS_PATH=./tf-approximate/tf2/examples/axmul_8x8/mul8u_
MULTS_SUFF=.bin

M1=${MULTS_PATH}${MULT1_NAME}${MULTS_SUFF}
M2=${MULTS_PATH}${MULT2_NAME}${MULTS_SUFF}
M3=${MULTS_PATH}${MULT3_NAME}${MULTS_SUFF}
M4=${MULTS_PATH}${MULT4_NAME}${MULTS_SUFF}
M5=${MULTS_PATH}${MULT5_NAME}${MULTS_SUFF}
#### -----------------------------------------------------------------------------

#### MODIFIABLE SETTINGS
WEIGHTS_NAME=myAlexNetWeights
TRAIN_EPOCHS=2

# Test multipliers ONLY for the purposes of the approx_eval target - can be edited
MULT1_NAME=EXZ
MULT2_NAME=125K
MULT3_NAME=2P7
MULT4_NAME=KEM
MULT5_NAME=ZFB
#### -----------------------------------------------------------------------------

# NSGA-II ALGORITHM SETTINGS
GENERATION_SIZE = 70
GENERATION_CNT = 20


approxLib:
	mkdir cnnInputs
	@echo "Downloading the libApproxGPUOpsTF.so compiled library!"
	wget --no-check-certificate -O './cnnInputs/libApproxGPUOpsTF.so' 'https://drive.google.com/uc?export=download&id=1FLB5HJ-qGM-akTcRWBglXN4pUokOWJXr'

train:
	@python testSetLoad.py
	@echo "Training AlexNet for 3 epochs......."
	@python train_AlexNet.py --weights ${WEIGHTS_NAME} --epochs ${TRAIN_EPOCHS}

multipliersReady:
	@echo "Cloning tf-approximate repository to have access to the multipliers......."
	@git clone https://github.com/ehw-fit/tf-approximate.git

std_eval:
	@echo "Evaluating AlexNet with standard convolution layers......."
	@python eval_AlexNet.py --weights ${WEIGHTS_NAME}

approx_eval:
	python eval_AlexNet.py --weights ${WEIGHTS_NAME} --fakeConv --m1 ${M1} --m2 ${M2} --m3 ${M3} --m4 ${M4} --m5 ${M5}

nsga2:
	@export TF_CPP_MIN_LOG_LEVEL="3"
	@echo "Running NSGA-II algorithm to determine the most efficient multipliers, generation size: ${GENERATION_SIZE}, generation count: ${GENERATION_CNT}"
	@python nsga2.py --weights ${WEIGHTS_NAME} --gen_size ${GENERATION_SIZE} --gen_cnt ${GENERATION_CNT}
	@export TF_CPP_MIN_LOG_LEVEL="1"

pack:
	zip -r xlanic04.zip Makefile *.py runsOutput README.md xlanic04.pdf
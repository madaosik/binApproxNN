##========== Copyright (c) 2021, Adam Lanicek, All rights reserved. =========##
##
## Purpose:     Setup module containing the energy consumption data for the
##              multipliers as well as providing and arra containing the
##              amount of multiplications in each corresponding layer.
##
## Implement.:  Implemented by Adam lanicek based on the information provided
##              at https://ehw.fit.vutbr.cz/evoapproxlib/
##
## $Date:       $2021-05-05
##============================================================================##

BIN_PATH = "../tf-approximate/tf2/examples/axmul_8x8/"
PREF = "mul8u_"
SUFF = ".bin"
MULT_NAME = 0
MULT_EN = 1
#[21  0 16 26 34]
mults = [
    ['125K', 0.384],
    ['12N4', 0.142],
    ['13QR', 0.0085],
    ['1446', 0.388],
    ['14VP', 0.364],
    ['150Q', 0.360],
    ['17C8', 0.0019],
    ['17KS', 0.104],
    ['17QU', 0.0017],
    ['185Q', 0.206],
    ['18DU', 0.031],
    #'199Z': 0,
    ['19DB', 0.206],
    ['1AGV', 0.095],
    ['1JFF', 0.391],
    ['2AC', 0.311],
    ['2HH', 0.302],
    ['2P7', 0.386],
    ['7C1', 0.329],
    ['96D', 0.309],
    ['CK5', 0.345],
    ['DM1', 0.195],
    ['EXZ', 0.380],
    ['FTA', 0.084],
    ['GS2', 0.356],
    ['JQQ', 0.371],
    ['JV3', 0.034],
    ['KEM', 0.370],
    ['L40', 0.189],
    ['NGR', 0.276],
    ['PKY', 0.254],
    ['QJD', 0.344],
    ['QKX', 0.029],
    ['Y48', 0.391],
    ['YX7', 0.061],
    ['ZFB', 0.304]
]

class ConvLayer:
    def __init__(self, inpSize, channelCnt, filterSize, outputChannels, strideSize):
        self.inpSize = inpSize
        self.channelCnt = channelCnt
        self.filterSize = filterSize
        self.outputChannels = outputChannels
        self.strideSize = strideSize
    
    def multOpsCnt(self):
        return int((self.inpSize * self.channelCnt * self.filterSize * self.outputChannels)/self.strideSize)

convLayersMult = []
conv1 = ConvLayer(inpSize=32*32, channelCnt=3, filterSize=11*11, outputChannels=96, strideSize=4*4)
convLayersMult.append(conv1.multOpsCnt())
conv2 = ConvLayer(inpSize=4*4, channelCnt=96, filterSize=5*5, outputChannels=256, strideSize=1*1)
convLayersMult.append(conv2.multOpsCnt())
conv3 = ConvLayer(inpSize=2*2, channelCnt=256, filterSize=3*3, outputChannels=384, strideSize=1*1)
convLayersMult.append(conv3.multOpsCnt())
conv4 = ConvLayer(inpSize=2*2, channelCnt=384, filterSize=3*3, outputChannels=384, strideSize=1*1)
convLayersMult.append(conv4.multOpsCnt())
conv5 = ConvLayer(inpSize=2*2, channelCnt=384, filterSize=3*3, outputChannels=256, strideSize=1*1)
convLayersMult.append(conv5.multOpsCnt())
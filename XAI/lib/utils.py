import os
import time
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import keras
import innvestigate
import xarray as xr
import pandas as pd
import datetime as dt
import glob
from tqdm import tqdm

XAI_PATH = './data/'

def getNeuronIdx(neuron):

    neuron = int(neuron)
    
    if neuron == 2000:
        return 23920
    elif neuron == 5950:
        return 2657
    else:
        print('Please provide a valid neuron index')
        return None

def normalizeSM(saliencyMaps):

    for idx in range(saliencyMaps.shape[0]):

        minSM = np.min(saliencyMaps[idx, :, :, :])
        maxSM = np.max(saliencyMaps[idx, :, :, :])
        saliencyMaps[idx, :, :, :] = (saliencyMaps[idx, :, :, :] - minSM) / (maxSM - minSM)

    return saliencyMaps

def compositeTrainSet(modelObj, modelName, neuronIdx, xData):

    # Get neuron index for CNNDoury
    if 'UNET' in modelName:
        neuron = getNeuronIdx(neuronIdx)
    else:
        neuron = neuronIdx

    # Set batch size for computing the SMs
    # Can't increase batchSize above 1 due to an existing bug in iNNvestigate
    # https://github.com/albermax/innvestigate/issues/246
    batchSize = 1

    # Pre-allocate memory for the saliencyMaps array
    saliencyMaps = np.empty(xData.shape)

    # Create analyzer
    analyzer = innvestigate.create_analyzer(name = 'integrated_gradients',
                                            model = modelObj,
                                            neuron_selection_mode = 'index')

    # First batch
    saliencyMaps[:batchSize, :] = analyzer.analyze(xData[:batchSize, :, :, :], neuron)
    saliencyMaps[:batchSize, :] = np.absolute(saliencyMaps[:batchSize, :])
    saliencyMaps[:batchSize, :] = normalizeSM(saliencyMaps[:batchSize, :])

    # Iterate over batches
    for i in tqdm(range(batchSize, xData.shape[0], batchSize)):
        
        saliencyMaps[i:i+batchSize, :] = analyzer.analyze(xData[i:i+batchSize, :, :, :], neuron)
        saliencyMaps[i:i+batchSize, :] = np.absolute(saliencyMaps[i:i+batchSize, :])
        saliencyMaps[i:i+batchSize, :] = normalizeSM(saliencyMaps[i:i+batchSize, :])

    # Save saliencyMaps as npy
    np.save(file = XAI_PATH + 'SMtrainSet_' + modelName + '_neuron' + str(neuronIdx) + '.npy',
            arr = saliencyMaps)
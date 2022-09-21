import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def normalizeList(data):
    data = np.array(data)
    data = (data) / (np.max(data) - np.min(data))
    return data

# PATHS
SD_PATH = '../statistical-downscaling/data/'
FIGS_PATH = './figs/'

# Read CSV
trainingLog_CNN10 = pd.read_csv(SD_PATH + 'trainLog_CNN10.csv', sep = ',')
trainingLog_CNN10_stand = pd.read_csv(SD_PATH + 'trainLog_CNN10_stand.csv', sep = ',')
trainingLog_CNNPan = pd.read_csv(SD_PATH + 'trainLog_CNNPan.csv', sep = ',')
trainingLog_CNN_UNET = pd.read_csv(SD_PATH + 'trainLog_CNN_UNET.csv', sep = ',')

# Filter on numEpoch
numEpoch = 10
trainingLog_CNN10 = trainingLog_CNN10[trainingLog_CNN10['epoch'] >= numEpoch]

trainingLog_CNN10_stand = trainingLog_CNN10_stand[trainingLog_CNN10_stand['epoch'] >= numEpoch]
trainingLog_CNN10_stand['loss'] = normalizeList(trainingLog_CNN10_stand['loss'])
trainingLog_CNN10_stand['val_loss'] = normalizeList(trainingLog_CNN10_stand['val_loss'])

trainingLog_CNNPan = trainingLog_CNNPan[trainingLog_CNNPan['epoch'] >= numEpoch]

trainingLog_CNN_UNET = trainingLog_CNN_UNET[trainingLog_CNN_UNET['epoch'] >= numEpoch]

# Compute plot
plt.figure(figsize = (10, 5))

plt.plot(trainingLog_CNN10['epoch'], trainingLog_CNN10['val_loss'],
         label = 'CNN10', c='red')

plt.plot(trainingLog_CNN10_stand['epoch'], trainingLog_CNN10_stand['val_loss'],
         label = 'CNN10-Stand', c='purple')

plt.plot(trainingLog_CNNPan['epoch'], trainingLog_CNNPan['val_loss'],
         label = 'CNNPan', c='blue')

plt.plot(trainingLog_CNN_UNET['epoch'], trainingLog_CNN_UNET['val_loss'],
         label = 'CNN-UNET', c='green')


plt.legend(title='Validation loss', fontsize=8)
plt.grid(alpha = 0.2)
plt.show()
plt.savefig(FIGS_PATH + 'figTrainLog_v2.pdf', bbox_inches='tight')
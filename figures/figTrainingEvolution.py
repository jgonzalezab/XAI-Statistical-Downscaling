import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# PATHS
SD_PATH = '../statistical-downscaling/data/'
FIGS_PATH = './figs/'

# Model to load history from
model = 'CNN10'

# Read CSV
trainingLog = pd.read_csv(SD_PATH + 'trainLog_' + model + '.csv', sep = ',')

# Filter on numEpoch
numEpoch = 10
trainingLog = trainingLog[trainingLog['epoch'] >= numEpoch]
 
# Compute plot
plt.figure(figsize = (10, 5))
plt.plot(trainingLog['epoch'], trainingLog['loss'], label = 'Training', c='red')
plt.plot(trainingLog['epoch'], trainingLog['val_loss'], label = 'Validation', c='blue')

plt.axvline(x=155, c = 'green', alpha = 0.5)
plt.text(185, 3.8, 'Opt', c = 'green', alpha = 0.5, rotation = 'vertical')

plt.axvline(x=800, c = 'green', alpha = 0.5)
plt.text(830, 3.8, '800', c = 'green', alpha = 0.5, rotation = 'vertical')

plt.axvline(x=3600, c = 'green', alpha = 0.5)
plt.text(3630, 3.8, '3600', c = 'green', alpha = 0.5, rotation = 'vertical')

plt.ylim(0, 4.3)

plt.legend()
plt.grid(alpha = 0.2)
plt.show()
plt.savefig(FIGS_PATH + 'figTrainLog.pdf', bbox_inches='tight')
import sys
import xarray as xr
import dask
import dask.array as da
import netCDF4
import numpy as np
from importlib import reload
import lib.loadData as loadData
import lib.utils as utils
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import datetime
import regionmask
import time
import warnings
warnings.filterwarnings("ignore")

# PATHS
DATA_PATH = '../preprocessData/data/'
DATA_METRICS_PATH = './data/'
FIGS_PATH = './figs/'

'''
[!]
This script loads data computed with computeMetricsGCM.py so if new data is
added re-run this script
[!]
'''

# Config params
metrics = ['P02', 'Mean', 'P98']

models = ['CNN10', 'CNN10_stand',
            'CNNPan', 'CNN_UNET', 'GCM']

namesPlot = {'CNN10': 'DeepESD',
                'CNN10_stand': 'DeepESD-Stand',
                'CNNPan': 'CNN-PAN',
                'CNN_UNET': 'CNN-UNET',
                'GCM': 'GCM'}

periods = ['2006-2040', '2041-2070', '2071-2100']
regions = ['NWN', 'NEN', 'WNA', 'CNA', 'ENA', 'NCA']

colors = {'GCM': 'black',
            'DeepESD': 'red',
            'DeepESD-Stand': 'blue',
            'CNN-PAN': 'green',
            'CNN-UNET': 'orange'}

yLimit = {'P02_2006-2040': (0, 13),
            'P02_2041-2070': (0, 13),
            'P02_2071-2100': (0, 13),
            'Mean_2006-2040': (0, 13),
            'Mean_2041-2070': (0, 13),
            'Mean_2071-2100': (0, 13),
            'P98_2006-2040': (0, 13),
            'P98_2041-2070': (0, 13),
            'P98_2071-2100': (0, 13)}

# Iterate over subfigures
i, j = 0, 0

# Initialize plot
fig, axs = plt.subplots(len(periods), len(metrics),
                        figsize = (16, 16),
                        squeeze = False)

for metric in metrics:

    # Iterate over columns
    for period in periods:

        # Initialize empty DF
        meanDF = utils.initDF([namesPlot.get(key) for key in models],
                                regions)

        # Iterate over figures
        for region in regions:
            for model in models:

                # Load data
                fileName = model + 'Delta_' + metric + '_' + period + '.npy'
                grid = np.load(DATA_METRICS_PATH + fileName)

                # Compute region's delta and assign to the DF
                meanDF.loc[(meanDF['Region'] == region) &
                            (meanDF['Model'] == namesPlot[model]), 'Delta'] = utils.computeRegionMean_TS(grid, region)

        groups = meanDF.groupby('Model')
        for name, group in groups:

            if i == 0:
                axs[i, j].set_title(metric, fontsize = 18)

            if j == 0:
                axs[i, j].set_ylabel(period, fontsize = 18)

            if name == 'GCM':
                axs[i, j].scatter(x = group['Region'], y = group['Delta'],
                                    label = name, vmin = 1, vmax = 10,
                                    marker = 'x', s = 300, c = colors[name],
                                    zorder = 20)
                axs[i, j].plot(group['Region'], group['Delta'],
                                c = colors[name], linestyle = '--',
                                lw = 1, zorder = 20)
            else:
                axs[i, j].scatter(x = group['Region'], y = group['Delta'],
                                    label = name, vmin = 1, vmax = 10,
                                    s = 20, c = colors[name])
                axs[i, j].plot(group['Region'], group['Delta'],
                                c = colors[name])

                axs[i, j].set_ylim(yLimit[metric + '_' + period])

            if (i, j) != (1, 0):
                for tick in axs[i, j].yaxis.get_major_ticks():
                    tick.tick1line.set_visible(False)
                    tick.tick2line.set_visible(False)
                    tick.label1.set_visible(False)
                    tick.label2.set_visible(False)

            if (i, j) != (2, 1):
                for tick in axs[i, j].xaxis.get_major_ticks():
                    tick.tick1line.set_visible(False)
                    tick.tick2line.set_visible(False)
                    tick.label1.set_visible(False)
                    tick.label2.set_visible(False)

            axs[i, j].grid(b = True, which = 'major', color = '#666666',
                            linestyle = '-', alpha = 0.5)
            axs[i, j].minorticks_on()

            axs[i, j].tick_params(axis = 'both', which = 'major', labelsize = 14)

        i = i + 1
    i = 0
    j = j + 1

handles, labels = axs[i-1, j-1].get_legend_handles_labels()
order = [4, 2, 3, 0, 1]

leg = fig.legend([handles[idx] for idx in order],
                    [labels[idx] for idx in order],
                    bbox_to_anchor = [0.35, 0.95],
                    loc = 'center', ncol = 3, fontsize = 18,
                    markerscale = 1.5)

plt.savefig(FIGS_PATH + 'figTS.pdf',
            dpi = 1000, bbox_inches = 'tight')
import sys
import xarray as xr
import dask
import dask.array as da
import netCDF4
import numpy as np
from importlib import reload
import matplotlib as mpl
from matplotlib import cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import regionmask
import cartopy.crs as ccrs
import warnings
warnings.filterwarnings("ignore")

# PATHS
SM_PATH = '../XAI/data/'
FIGS_PATH = './figs/'

# IPCC Regions
regions = ['NWN', 'NEN', 'WNA', 'CNA', 'ENA', 'NCA']
regionsToRemove = ['SCA', 'CAR']

# Models
models = ['CNN10', 'CNN10_stand',
          'CNNPan', 'CNN_UNET']

namesPlot = {'CNN10': 'DeepESD',
             'CNN10_stand': 'DeepESD-Stand',
             'CNNPan': 'CNN-PAN',
             'CNN_UNET': 'CNN-UNET'}

# Neurons
idxNeuron = ['2000', '5950']

# Vars to plot
varsToPlot = ['ta@1000', 'hus@1000']

# Vars and heights dicts
allVars = ['z@500', 'z@700', 'z@850', 'z@1000',
           'hus@500', 'hus@700', 'hus@850', 'hus@1000',
           'ta@500', 'ta@700', 'ta@850', 'ta@1000',
           'ua@500', 'ua@700', 'ua@850', 'ua@1000',
           'va@500', 'va@700', 'va@850', 'va@1000']

# Initialize figure
nRows = len(models)

fig = plt.figure(figsize = (30, 20))
outer = gridspec.GridSpec(nRows, 1,
                          wspace = -0.9, hspace = -0.1)

importanceMin = 0.1
importanceMax = 0.7
colorSchema = 'magma_r'
cmap = plt.get_cmap(colorSchema)
cmap.set_under('white')

# IPCC Regions
ar6 = regionmask.defined_regions.ar6.all
northAmerica = ar6[regions]

# Dict defining coords to print region mean values
dictCoord = {'NWN': [0.4, 0.85],
             'NEN': [0.8, 0.85],
             'WNA': [0.45, 0.55],
             'CNA': [0.65, 0.55],
             'ENA': [0.8, 0.5],
             'NCA': [0.57, 0.25],
             'SCA': [0.69, 0.1],
             'CAR': [0.89, 0.1]}

# Iterate over subplots
generalRow = 0
for model in models:

    # Initialize inner plot
    inner = gridspec.GridSpecFromSubplotSpec(1, len(varsToPlot) * len(idxNeuron),
                                             subplot_spec = outer[generalRow],
                                             wspace = 0.1, hspace = 0.1)

    # Plot Saliency Maps
    innerCol = 0

    for var in varsToPlot:
        for neuron in idxNeuron:

            # Load Saliency Map
            nameSM =  'SMtrainSet_' + model + '_neuron' + str(neuron) + '.npy'
            saliencyMaps = np.load(SM_PATH + nameSM)

            # Compute mean of saliency maps
            saliencyMaps = np.mean(saliencyMaps, axis=0)

            # Idx of variable
            combIdx = allVars.index(var)

            # Inner
            axes = plt.Subplot(fig, inner[0, innerCol])

            if innerCol == 0:
                axes.set_ylabel(namesPlot[model], fontsize = 16, weight = 'bold')

            # Compute subplot
            map = Basemap(ax = axes,
                          llcrnrlon = -164.75, llcrnrlat = 11.75,
                          urcrnrlon = -59.75, urcrnrlat = 69.75,
                          resolution = 'c')

            im = map.imshow(saliencyMaps[:, :, combIdx],
                            vmin = importanceMin, vmax = importanceMax,
                            cmap = cmap)

            map.drawcoastlines(linewidth = 0.2, color = 'gray')

            fig.add_subplot(axes)

            innerCol = innerCol + 1

    # Saliency maps colorbar
    cbar_ax = fig.add_axes([0.36, 0.08, 0.3, 0.012])
    cb = fig.colorbar(im, cax = cbar_ax, orientation = 'horizontal',
                      extend = 'min')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.tick_params(labelsize = 24)
    cb.set_label(label = 'Relevance (unitless)', fontsize = 26)

    generalRow = generalRow + 1

# Cols title
fig.text(0.255-0.055, 0.9, 'Air temperature (1000 hPa)',
         fontsize = 36)
fig.text(0.175, 0.86, 'North Point',
         fontsize = 28)
fig.text(0.375, 0.86, 'South Point',
         fontsize = 28)

fig.text(0.65-0.065, 0.9, 'Specific Humidity (1000 hPa)',
         fontsize = 36)
fig.text(0.57, 0.86, 'North Point',
         fontsize = 28)
fig.text(0.77, 0.86, 'South Point',
         fontsize = 28)

plt.savefig(FIGS_PATH + 'figSM_train.pdf',
            dpi = 300, bbox_inches = 'tight')
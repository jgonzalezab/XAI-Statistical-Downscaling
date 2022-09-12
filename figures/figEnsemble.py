import sys
import xarray as xr
import dask
import dask.array as da
import netCDF4
import numpy as np
from importlib import reload
import lib.loadData as loadData
import lib.utils as utils
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib import cm
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
DATA_PATH = '../preprocessData/data/'
DATA_METRICS_PATH = './data/'
FIGS_PATH = './figs/'

# IPCC Regions
regions = ['NWN', 'NEN', 'WNA', 'CNA', 'ENA', 'NCA']
regionsToRemove = ['SCA', 'CAR']

# Periods
yearsPeriods = ['1980-2002', '2006-2040', '2041-2070', '2071-2100']

# Models
models = ['CNN10', 'CNN10_stand',
            'CNNPan', 'CNN_UNET']

namesPlot = {'CNN10': 'DeepESD',
                'CNN10_stand': 'DeepESD-Stand',
                'CNNPan': 'CNN-PAN',
                'CNN_UNET': 'CNN-UNET'}

metrics = ['Bias P02',
            'Bias Mean',
            'Bias P98']

namesMetrics = {'Bias P02': 'P02',
                'Bias Mean': 'Mean',
                'Bias P98': 'P98'}

# Custom color bar
cmapEnsemble = cm.get_cmap('Reds', 9)
boundsEnsemble = list(range(0, 10))
normEnsemble = mpl.colors.BoundaryNorm(boundsEnsemble, cmapEnsemble.N)

cmapSD = cm.get_cmap('Blues', 9)

# Load GCM Projection NetCDF
yGCMHist = loadData.loadHistProj(years = slice('1980-01-01', '2002-12-31'))
yGCMHist = yGCMHist.assign_coords(time = pd.to_datetime(yGCMHist.time.values).normalize())

yGCMFuture_1 = loadData.loadFutureProj(years = slice('2006-01-01', '2040-12-31'))
yGCMFuture_1 = yGCMFuture_1.assign_coords(time = pd.to_datetime(yGCMFuture_1.time.values).normalize())

yGCMFuture_2 = loadData.loadFutureProj(years = slice('2041-01-01', '2070-12-31'))
yGCMFuture_2 = yGCMFuture_2.assign_coords(time = pd.to_datetime(yGCMFuture_2.time.values).normalize())

yGCMFuture_3 = loadData.loadFutureProj(years = slice('2071-01-01', '2100-12-31'))
yGCMFuture_3 = yGCMFuture_3.assign_coords(time = pd.to_datetime(yGCMFuture_3.time.values).normalize())

# Compute GCM Deltas
for years in yearsPeriods[1:]:

    if years == '2006-2040':
        compMetric = utils.BiasMean(y = yGCMHist, x = yGCMFuture_1)
        compMetric = utils.maskRegions(compMetric, regionsToRemove)
        compMetric = utils.applyMask(grid = compMetric)
        deltaGCM_Future_1 = compMetric
    elif years == '2041-2070':
        compMetric = utils.BiasMean(y = yGCMHist, x = yGCMFuture_2)
        compMetric = utils.maskRegions(compMetric, regionsToRemove)
        compMetric = utils.applyMask(grid = compMetric)
        deltaGCM_Future_2 = compMetric
    elif years == '2071-2100':
        compMetric = utils.BiasMean(y = yGCMHist, x = yGCMFuture_3)
        compMetric = utils.maskRegions(compMetric, regionsToRemove)
        compMetric = utils.applyMask(grid = compMetric)
        deltaGCM_Future_3 = compMetric

# Compute models ensemble
yEnsembleHist = loadData.loadHistoricalPredProj(model = models[0],
                                                years = slice('1980-01-01', '2002-12-31'))

yEnsembleFuture_1 = loadData.loadFuturePredProj(model = models[0],
                                                years = slice('2006-01-01', '2040-12-31'))

yEnsembleFuture_2 = loadData.loadFuturePredProj(model = models[0],
                                                years = slice('2041-01-01', '2070-12-31'))

yEnsembleFuture_3 = loadData.loadFuturePredProj(model = models[0],
                                                years = slice('2071-01-01', '2100-12-31'))

for years in yearsPeriods:

    if years == '1980-2002':
        for model in models[1:]:
            aux = loadData.loadHistoricalPredProj(model = model,
                                                  years = slice('1980-01-01', '2002-12-31'))
            yEnsembleHist = yEnsembleHist + aux

    elif years == '2006-2040':
        for model in models[1:]:
            aux = loadData.loadFuturePredProj(model = model,
                                              years = slice('2006-01-01', '2040-12-31'))
            yEnsembleFuture_1 = yEnsembleFuture_1 + aux

    elif years == '2041-2070':
        for model in models[1:]:
            aux = loadData.loadFuturePredProj(model = model,
                                              years = slice('2041-01-01', '2070-12-31'))
            yEnsembleFuture_2 = yEnsembleFuture_2 + aux

    elif years == '2071-2100':
        for model in models[1:]:
            aux = loadData.loadFuturePredProj(model = model,
                                              years = slice('2071-01-01', '2100-12-31'))
            yEnsembleFuture_3 = yEnsembleFuture_3 + aux

yEnsembleHist = yEnsembleHist / len(models)
yEnsembleFuture_1 = yEnsembleFuture_1 / len(models)
yEnsembleFuture_2 = yEnsembleFuture_2 / len(models)
yEnsembleFuture_3 = yEnsembleFuture_3 / len(models)

# Compute ensemble deltas
for years in yearsPeriods[1:]:

    if years == '2006-2040':
        compMetric = utils.BiasMean(y = yEnsembleHist, x = yEnsembleFuture_1)
        compMetric = utils.maskRegions(compMetric, regionsToRemove)
        compMetric = utils.applyMask(grid = compMetric)
        deltaEnsemble_Future_1 = compMetric
    elif years == '2041-2070':
        compMetric = utils.BiasMean(y = yEnsembleHist, x = yEnsembleFuture_2)
        compMetric = utils.maskRegions(compMetric, regionsToRemove)
        compMetric = utils.applyMask(grid = compMetric)
        deltaEnsemble_Future_2 = compMetric
    elif years == '2071-2100':
        compMetric = utils.BiasMean(y = yEnsembleHist, x = yEnsembleFuture_3)
        compMetric = utils.maskRegions(compMetric, regionsToRemove)
        compMetric = utils.applyMask(grid = compMetric)
        deltaEnsemble_Future_3 = compMetric

# Compute standard deviation of the ensemble
# (Standard deviation between deltas of the model used in the ensemble)
modelsDeltas = {}

for model in models:

    yGCMHistPred = loadData.loadHistoricalPredProj(model = model,
                                                   years = slice('1980-01-01', '2002-12-31'))

    yGCMFuturePred_1 = loadData.loadFuturePredProj(model = model,
                                                   years = slice('2006-01-01', '2040-12-31'))

    yGCMFuturePred_2 = loadData.loadFuturePredProj(model = model,
                                                   years = slice('2041-01-01', '2070-12-31'))

    yGCMFuturePred_3 = loadData.loadFuturePredProj(model = model,
                                                   years = slice('2071-01-01', '2100-12-31'))

    for years in yearsPeriods[1:]:

        if years == '2006-2040':
            compMetric = utils.BiasMean(y = yGCMHistPred, x = yGCMFuturePred_1)
            compMetric = utils.maskRegions(compMetric, regionsToRemove)
            compMetric = utils.applyMask(grid = compMetric)
            modelsDeltas[model + '_' + years] = compMetric
        elif years == '2041-2070':
            compMetric = utils.BiasMean(y = yGCMHistPred, x = yGCMFuturePred_2)
            compMetric = utils.maskRegions(compMetric, regionsToRemove)
            compMetric = utils.applyMask(grid = compMetric)
            modelsDeltas[model + '_' + years] = compMetric
        elif years == '2071-2100':
            compMetric = utils.BiasMean(y = yGCMHistPred, x = yGCMFuturePred_3)
            compMetric = utils.maskRegions(compMetric, regionsToRemove)
            compMetric = utils.applyMask(grid = compMetric)
            modelsDeltas[model + '_' + years] = compMetric

# Future_1 SD
ensembleSD_future_1 = (modelsDeltas[models[0] + '_' + '2006-2040'] - deltaEnsemble_Future_1) ** 2
for model in models[1:]:
    ensembleSD_future_1 = ensembleSD_future_1 + ((modelsDeltas[model + '_' + '2006-2040'] - deltaEnsemble_Future_1) ** 2)
ensembleSD_future_1 = np.sqrt(ensembleSD_future_1 / len(models))

# Future_2 SD
ensembleSD_future_2 = (modelsDeltas[models[0] + '_' + '2041-2070'] - deltaEnsemble_Future_2) ** 2
for model in models[1:]:
    ensembleSD_future_2 = ensembleSD_future_2 + ((modelsDeltas[model + '_' + '2041-2070'] - deltaEnsemble_Future_2) ** 2)
ensembleSD_future_2 = np.sqrt(ensembleSD_future_2 / len(models))

# Future_3 SD
ensembleSD_future_3 = (modelsDeltas[models[0] + '_' + '2071-2100'] - deltaEnsemble_Future_3) ** 2
for model in models[1:]:
    ensembleSD_future_3 = ensembleSD_future_3 + ((modelsDeltas[model + '_' + '2071-2100'] - deltaEnsemble_Future_3) ** 2)
ensembleSD_future_3 = np.sqrt(ensembleSD_future_3 / len(models))

# Initialize figure
fig, axes = plt.subplots(3, 3, figsize = (12, 12))

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


# Plot subfigures i, 0
for i in range(3):

    if i == 0:
        compMetric = deltaGCM_Future_1
    elif i == 1:
        compMetric = deltaGCM_Future_2
    elif i == 2:
        compMetric = deltaGCM_Future_3

    axes[i, 0] = northAmerica.plot(ax = axes[i, 0],
                                    add_ocean = False,
                                    add_label = False,
                                    line_kws = dict(linewidth = 1),
                                    coastlines = False)

    dictMeans =  utils.computeRegionMeans(compMetric, regions)
    for region in regions:
        axes[i, 0].text(dictCoord[region][0],
                        dictCoord[region][1],
                        s = r'$\bf{' + str(dictMeans[region]) + '}$',
                        fontsize = 10,
                        horizontalalignment = 'center',
                        verticalalignment = 'center',
                        transform = axes[i, 0].transAxes)

    axes[i, 0].text(0.18, 0.5,
                    s = r'$\bf{' + str(np.round(utils.mean(compMetric).compute(), 2)) + '}$',
                    fontsize = 18,
                    horizontalalignment = 'center',
                    verticalalignment = 'center',
                    transform = axes[i, 0].transAxes)

    if i == 0:
        axes[i, 0].set_title('EC-Earth Delta Change',
                                fontsize = 18)
    axes[i, 0].set_aspect('equal')
    map = Basemap(ax = axes[i, 0],
                    llcrnrlon = -164.75, llcrnrlat = 11.75,
                    urcrnrlon = -59.75, urcrnrlat = 69.75,
                    resolution = 'c')
    map.drawcoastlines(linewidth = 0.6)

    im = map.imshow(compMetric, vmin = 0, vmax = 10,
                    cmap = cmapEnsemble)

# Plot subfigures i, 1
for i in range(3):

    if i == 0:
        compMetric = deltaEnsemble_Future_1
    elif i == 1:
        compMetric = deltaEnsemble_Future_2
    elif i == 2:
        compMetric = deltaEnsemble_Future_3

    axes[i, 1] = northAmerica.plot(ax = axes[i, 1],
                                    add_ocean = False,
                                    add_label = False,
                                    line_kws = dict(linewidth = 1),
                                    coastlines = False)

    dictMeans =  utils.computeRegionMeans(compMetric, regions)
    for region in regions:
        axes[i, 1].text(dictCoord[region][0],
                        dictCoord[region][1],
                        s = r'$\bf{' + str(dictMeans[region]) + '}$',
                        fontsize = 10,
                        horizontalalignment = 'center',
                        verticalalignment = 'center',
                        transform = axes[i, 1].transAxes)

    axes[i, 1].text(0.18, 0.5,
                    s = r'$\bf{' + str(np.round(utils.mean(compMetric).compute(), 2)) + '}$',
                    fontsize = 18,
                    horizontalalignment = 'center',
                    verticalalignment = 'center',
                    transform = axes[i, 1].transAxes)

    if i == 0:
        axes[i, 1].set_title('Ensemble Delta Change',
                                fontsize = 18)
    axes[i, 1].set_aspect('equal')
    map = Basemap(ax = axes[i, 1],
                    llcrnrlon = -164.75, llcrnrlat = 11.75,
                    urcrnrlon = -59.75, urcrnrlat = 69.75,
                    resolution = 'c')
    map.drawcoastlines(linewidth = 0.6)

    im = map.imshow(compMetric, vmin = 0, vmax = 10,
                    cmap = cmapEnsemble)

cax = fig.add_axes([0.2, 0.4, 0.4, 0.02])
cb = mpl.colorbar.ColorbarBase(cax, cmap = cmapEnsemble,
                                norm = normEnsemble,
                                ticks = boundsEnsemble,
                                orientation = 'horizontal')
cb.ax.tick_params(labelsize = 16)

# Plot subfigures i, 2
for i in range(3):

    if i == 0:
        compMetric = ensembleSD_future_1
    elif i == 1:
        compMetric = ensembleSD_future_2
    elif i == 2:
        compMetric = ensembleSD_future_3

    axes[i, 2] = northAmerica.plot(ax = axes[i, 2],
                                    add_ocean = False,
                                    add_label = False,
                                    line_kws = dict(linewidth = 1),
                                    coastlines = False)

    dictMeans =  utils.computeRegionMeans(compMetric, regions)
    for region in regions:
        axes[i, 2].text(dictCoord[region][0],
                        dictCoord[region][1],
                        s = r'$\bf{' + str(dictMeans[region]) + '}$',
                        fontsize = 10,
                        horizontalalignment = 'center',
                        verticalalignment = 'center',
                        transform = axes[i, 2].transAxes)

    axes[i, 2].text(0.18, 0.5,
                    s = r'$\bf{' + str(np.round(utils.mean(compMetric).compute(), 2)) + '}$',
                    fontsize = 18,
                    horizontalalignment = 'center',
                    verticalalignment = 'center',
                    transform = axes[i, 2].transAxes)

    if i == 0:
        axes[i, 2].set_title('Standard dev. of ensemble',
                                fontsize = 18)
    axes[i, 2].set_aspect('equal')
    map = Basemap(ax = axes[i, 2],
                    llcrnrlon = -164.75, llcrnrlat = 11.75,
                    urcrnrlon = -59.75, urcrnrlat = 69.75,
                    resolution = 'c')
    map.drawcoastlines(linewidth = 0.6)

    im = map.imshow(compMetric, vmin = 0, vmax = 2,
                    cmap = cmapSD)

    cax = fig.add_axes([0.75, 0.4, 0.2, 0.02])
    cb = plt.colorbar(im, cax = cax, orientation = 'horizontal')
    cb.ax.tick_params(labelsize = 16)


    axes[0, 0].set_ylabel('2006-2040', fontsize = 16)
    axes[1, 0].set_ylabel('2041-2070', fontsize = 16)
    axes[2, 0].set_ylabel('2071-2100', fontsize = 16)

plt.tight_layout(rect = [0.08, 0.38, 1, 1])
plt.savefig(FIGS_PATH + 'figEnsemble.pdf',
            dpi = 1000, bbox_inches = 'tight')
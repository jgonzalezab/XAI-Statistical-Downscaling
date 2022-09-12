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

# Models
models = ['CNN10', 'CNN10_stand',
            'CNNPan', 'CNN_UNET']

namesPlot = {'CNN10': 'DeepESD',
                'CNN10_stand': 'DeepESD-Stand',
                'CNNPan': 'CNN-PAN',
                'CNN_UNET': 'CNN-UNET'}

metrics = ['Bias P02',
            'Bias Mean',
            'Bias P98',
            'RMSE']

# Load target NetCDF
yTarget = loadData.loadTarget(years = slice('2003-01-01', '2008-12-31'))
yTarget = yTarget.assign_coords(time = pd.to_datetime(yTarget.time.values).normalize())

# Initialize figure
fig = plt.figure(figsize = (20, 3 * len(models)))
outer = gridspec.GridSpec(len(models), 1)

# Colors for RMSE
cmapRMSE = cm.get_cmap('Reds', 9)
boundsRMSE = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
normRMSE = mpl.colors.BoundaryNorm(boundsRMSE, cmapRMSE.N)
ticksRMSE = [0, 1, 2, 3, 4,]

# Custom color bar for biases
customColors = ["#67001F", "#B2182B", "#D6604D", "#F4A582", "#FFFFFF", "#FFFFFF",
                "#FFFFFF", "#92C5DE", "#4393C3", "#2166AC", "#053061"]
customColors = customColors[::-1]

cmapBias = mpl.colors.ListedColormap(customColors)
boundsBias = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
normBias = mpl.colors.BoundaryNorm(boundsBias, cmapBias.N)
ticksBiases = [-2, -1, 0, 1, 2]

# Colors for Relative RMSE
cmapSD = cm.get_cmap('Purples', 13)
minSD = 0; maxSD = 0.5
boundsSD = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
normSD = mpl.colors.BoundaryNorm(boundsSD, cmapSD.N)
ticksSD = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

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

# Iterate over models (rows)
generalRow = 0
for model in models:

    print(model + ':')

    # Initialize inner plot
    inner = gridspec.GridSpecFromSubplotSpec(1, 4, 
                                                subplot_spec = outer[generalRow],
                                                wspace = -0.15) 

    # Load predictions
    yPred = loadData.loadTestPred(model = model)
    yPred = yPred.assign_coords(time = pd.to_datetime(yPred.time.values).normalize())

    # Iterate over metrics (columns)
    for metric in metrics:

        print(metric)

        if metric == 'Bias P02':

            yTargetP02 = da.apply_along_axis(utils.P02, 0, yTarget.tas)
            yPredP02 = da.apply_along_axis(utils.P02, 0, yPred.tas)
            P02Bias = yPredP02 - yTargetP02
            P02BiasAbs = np.abs(yPredP02 - yTargetP02)
            compMetric = P02Bias.compute()
            compMetricAbs = P02BiasAbs.compute()
            i = 0
            j = 0

        elif metric == 'Bias Mean':

            yTargetMean = da.apply_along_axis(utils.mean, 0, yTarget.tas)
            yPredMean = da.apply_along_axis(utils.mean, 0, yPred.tas)
            meanBias = yPredMean - yTargetMean
            meanBiasAbs = np.abs(yPredMean - yTargetMean)
            compMetric = meanBias.compute()
            compMetricAbs = meanBiasAbs.compute()
            i = 0
            j = 1

        elif metric == 'Bias P98':

            yTargetP98 = da.apply_along_axis(utils.P98, 0, yTarget.tas)
            yPredP98 = da.apply_along_axis(utils.P98, 0, yPred.tas)
            P98Bias = yPredP98 - yTargetP98
            P98BiasAbs = np.abs(yPredP98 - yTargetP98)
            compMetric = P98Bias.compute()
            compMetricAbs = P98BiasAbs.compute()
            i = 0
            j = 2

        elif metric == 'RMSE':

            yTargetAUX = yTarget.sel(time = pd.to_datetime(yPred.time.values))
            rmse = ((yPred - yTarget) ** 2)
            rmse = da.apply_along_axis(utils.mean, 0, rmse.tas)
            rmse = rmse ** (1/2)
            compMetric = rmse.compute()
            i = 0
            j = 3

        ####################################################################
        compMetric = utils.maskRegions(compMetric, regionsToRemove)
        compMetric = utils.applyMask(grid = compMetric)
        ####################################################################

        axes = plt.Subplot(fig, inner[i, j])

        axes = northAmerica.plot(ax = axes,
                                    add_ocean = False,
                                    add_label = False,
                                    line_kws = dict(linewidth = 1),
                                    coastlines = False)

        dictMeans = utils.computeRegionMeans(compMetricAbs, regions)
        for region in regions:
            axes.text(dictCoord[region][0],
                        dictCoord[region][1],
                        s = r'$\bf{' + str(dictMeans[region]) + '}$',
                        fontsize = 12,
                        horizontalalignment = 'center',
                        verticalalignment = 'center',
                        transform = axes.transAxes)

        if generalRow == 0:
            axes.set_title(metric,
                            fontsize = 18)

        axes.set_aspect('equal')
        map = Basemap(ax = axes,
                        llcrnrlon = -164.75, llcrnrlat = 11.75,
                        urcrnrlon = -59.75, urcrnrlat = 69.75,
                        resolution = 'c')
        map.drawcoastlines(linewidth = 0.6)

        if metric == 'RMSE':
            im = map.imshow(compMetric, vmin = 0, vmax = 4,
                            cmap = cmapRMSE, norm = normRMSE)

            if generalRow == len(models)-1:
                cax = fig.add_axes([0.71, 0.08, 0.16, 0.02])
                cb = plt.colorbar(im, cax = cax, orientation = 'horizontal',
                                    ticks = ticksRMSE)
                cb.ax.tick_params(labelsize = 12)

            axes.text(0.18, 0.5,
                        s = r'$\bf{' + str(np.round(utils.mean(compMetric), 2).compute()) + '}$',
                        fontsize = 16,
                        horizontalalignment = 'center',
                        verticalalignment = 'center',
                        transform = axes.transAxes)

        elif metric == 'Relative RMSE':
            im = map.imshow(compMetric, vmin = 0, vmax = 20,
                            cmap = cmapSD, norm = normSD)

            axes.text(0.18, 0.5,
                        s = r'$\bf{' + str(np.round(utils.mean(compMetric), 2).compute()) + '}$',
                        fontsize = 16,
                        horizontalalignment = 'center',
                        verticalalignment = 'center',
                        transform = axes.transAxes)

        else:
            im = map.imshow(compMetric, vmin = -2, vmax = 2,
                            cmap = cmapBias, norm = normBias)

            axes.text(0.18, 0.5,
                        s = r'$\bf{' + str(np.round(utils.mean(compMetricAbs), 2)) + '}$',
                        fontsize = 16,
                        horizontalalignment = 'center',
                        verticalalignment = 'center',
                        transform = axes.transAxes)

            if metric == 'Bias Mean' and generalRow == len(models)-1:
                cax = fig.add_axes([0.31, 0.08, 0.22, 0.02])
                cb = plt.colorbar(im, cax = cax, orientation = 'horizontal')
                cb.ax.tick_params(labelsize = 12)

        fig.add_subplot(axes)

    generalRow += 1
    print('\n')

fig.text(0.13, 0.75, 'DeepESD', weight = 'bold',
            fontsize = 18, rotation = 90)
fig.text(0.13, 0.51, 'DeepESD-Stand', weight = 'bold',
            fontsize = 18, rotation = 90)
fig.text(0.13, 0.35, 'CNN-PAN', weight = 'bold',
            fontsize = 18, rotation = 90)
fig.text(0.13, 0.14, 'CNN-UNET', weight = 'bold',
            fontsize = 18, rotation = 90)

plt.savefig(FIGS_PATH + 'fig-ValidMetrics.pdf',
            dpi = 1000, bbox_inches = 'tight')
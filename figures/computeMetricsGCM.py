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
regions = ['NWN', 'NEN', 'WNA', 'CNA', 'ENA', 'NCA', 'SCA', 'CAR']
regionsToRemove = ['SCA', 'CAR']

# Periods 
periods = ['2006-2040', '2041-2070',  '2071-2100']

for period in periods:

    print('')
    print(period)

    if period == '2006-2040':
        yearsGCM = slice('2006-01-01', '2040-12-31')
    elif period == '2041-2070' :
        yearsGCM = slice('2041-01-01', '2070-12-31')
    elif period == '2071-2100':
        yearsGCM = slice('2071-01-01', '2100-12-31')

    models = ['CNN10', 'CNN10_stand',
                'CNNPan', 'CNN_UNET']

    metrics = ['Bias P02',
                'Bias Mean',
                'Bias P98']

    # Load GCM Projection NetCDF
    yGCMHist = loadData.loadHistProj(years = slice('1980-01-01', '2002-12-31'))
    yGCMHist = yGCMHist.assign_coords(time = pd.to_datetime(yGCMHist.time.values).normalize())

    yGCMFuture = loadData.loadFutureProj(years = yearsGCM)
    yGCMFuture = yGCMFuture.assign_coords(time = pd.to_datetime(yGCMFuture.time.values).normalize())

    # GCM Deltas
    print('GCM:')
    for metric in metrics:

        print(metric)

        if metric == 'Bias P02':
            compMetric = utils.BiasP02(y = yGCMHist, x = yGCMFuture)
            compMetric = utils.applyMask(grid = compMetric)
            GCMP02 = compMetric
            np.save(DATA_METRICS_PATH + 'GCMDelta_P02_' + period + '.npy',
                    np.array(compMetric))
        elif metric == 'Bias Mean':
            compMetric = utils.BiasMean(y = yGCMHist, x = yGCMFuture)
            compMetric = utils.applyMask(grid = compMetric)
            GCMMean = compMetric
            np.save(DATA_METRICS_PATH + 'GCMDelta_Mean_' + period + '.npy',
                    np.array(compMetric))
        elif metric == 'Bias P98':
            compMetric = utils.BiasP98(y = yGCMHist, x = yGCMFuture)
            compMetric = utils.applyMask(grid = compMetric)
            GCMP98 = compMetric
            np.save(DATA_METRICS_PATH + 'GCMDelta_P98_' + period + '.npy',
                    np.array(compMetric))

    for model in models:

        print(model + ':')

        yGCMHistPred = loadData.loadHistoricalPredProj(years = slice('1980-01-01', '2002-12-31'),
                                                        model=model)
        yGCMHistPred = yGCMHistPred.assign_coords(time = pd.to_datetime(yGCMHistPred.time.values).normalize())

        yGCMFuturePred = loadData.loadFuturePredProj(model = model,
                                                        years = yearsGCM)
        yGCMFuturePred = yGCMFuturePred.assign_coords(time = pd.to_datetime(yGCMFuturePred.time.values).normalize())

        for metric in metrics:

            print(metric)

            if metric == 'Bias P02':
                compMetric = utils.BiasP02(y = yGCMHistPred, x = yGCMFuturePred)
                compMetric = utils.applyMask(grid = compMetric) 
                modelP02 = compMetric
                np.save(DATA_METRICS_PATH + model + 'Delta_P02_' + period + '.npy',
                        np.array(compMetric))
            elif metric == 'Bias Mean':
                compMetric = utils.BiasMean(y = yGCMHistPred, x = yGCMFuturePred)
                compMetric = utils.applyMask(grid = compMetric) 
                modelMean = compMetric
                np.save(DATA_METRICS_PATH + model + 'Delta_Mean_' + period + '.npy',
                        np.array(compMetric))
            elif metric == 'Bias P98':
                compMetric = utils.BiasP98(y = yGCMHistPred, x = yGCMFuturePred)
                compMetric = utils.applyMask(grid = compMetric) 
                modelP98 = compMetric
                np.save(DATA_METRICS_PATH + model + 'Delta_P98_' + period + '.npy',
                        np.array(compMetric))
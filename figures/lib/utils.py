import xarray as xr
import dask
import dask.array as da
import netCDF4
import numpy as np
from importlib import reload
import lib.loadData as loadData
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import regionmask
import cartopy.crs as ccrs

# PATHS
DATA_PATH = '../preprocessData/data/'
DATA_METRICS_PATH = './data/'
FIGS_PATH = './figs/'

def P02(x):
        return np.nanquantile(x, 0.02)

def mean(x):
    return np.nanmean(x)

def P98(x):
        return np.nanquantile(x, 0.98)

def BiasP02(y, x):

    def P02(x):
        return np.nanquantile(x, 0.02)

    yP02 = da.apply_along_axis(P02, 0, y.tas)
    xP02 = da.apply_along_axis(P02, 0, x.tas)
    P02Bias = xP02 - yP02
    compMetric = P02Bias.compute()

    return compMetric

def BiasMean(y, x):

    def mean(x):
        return np.nanmean(x)

    yMean = da.apply_along_axis(mean, 0, y.tas)
    xMean = da.apply_along_axis(mean, 0, x.tas)
    MeanBias = xMean - yMean
    compMetric = MeanBias.compute()

    return compMetric

def BiasP98(y, x):

    def P98(x):
        return np.nanquantile(x, 0.98)

    yP98 = da.apply_along_axis(P98, 0, y.tas)
    xP98 = da.apply_along_axis(P98, 0, x.tas)
    P98Bias = xP98 - yP98
    compMetric = P98Bias.compute()

    return compMetric

def standDev(x):
    return np.nanstd(x)

def applyMask(grid):

    mask = xr.open_dataset(DATA_PATH + 'yMask.nc4', chunks = None)

    mask = (mask - mask) + 1

    mask = np.isnan(mask.tas)
    mask = mask[0, :, :]

    maskedGrid = da.ma.masked_array(grid, mask = mask)
    return maskedGrid

def applyMaskNumpy(grid):

    mask = xr.open_dataset(DATA_PATH + 'yMask.nc4', chunks = None)

    mask = (mask - mask) + 1

    mask = np.array(mask.tas)
    mask = mask[0, :, :]

    grid = np.array(grid)

    maskedGrid = grid * mask
    return maskedGrid

def maskRegions(grid, regions):

    y = loadData.loadTarget(years = slice('1980-01-01', '1980-01-02'))

    ar6 = regionmask.defined_regions.ar6.all
    mask = ar6.mask_3D(y.tas)

    for region in regions:

        regionMask = mask.isel(region = (mask.abbrevs == region))
        grid = da.where(regionMask, np.nan, grid)

    grid = da.squeeze(grid)

    return grid

def computeRegionMeans(grid, regions):

    y = loadData.loadTarget(years = slice('1980-01-01', '1980-01-02'))
    dictMeans = {}

    ar6 = regionmask.defined_regions.ar6.all
    mask = ar6.mask_3D(y.tas)

    for region in regions:

        regionMask = mask.isel(region = (mask.abbrevs == region))
        gridMasked = da.where(regionMask, grid, np.nan)

        regionMean = da.nanmean(gridMasked).compute()
        dictMeans[region] = np.round(regionMean, 2)

    return dictMeans

def initDF(models, regions):

    regionsRepeated = np.repeat(regions, len(models))
    modelsRepeated = models * len(regions)

    finalDF = pd.DataFrame({'Region': regionsRepeated,
                            'Model': modelsRepeated,
                            'Delta': np.nan})

    return finalDF

def computeRegionMean_TS(grid, region):

    y = loadData.loadTarget(years = slice('1980-01-01', '1980-01-02'))

    ar6 = regionmask.defined_regions.ar6.all
    mask = ar6.mask_3D(y.tas)

    regionMask = mask.isel(region = (mask.abbrevs == region))
    gridMasked = np.where(regionMask, grid, np.nan)

    regionMean = np.nanmean(gridMasked)

    return regionMean   
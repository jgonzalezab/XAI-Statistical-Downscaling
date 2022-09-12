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

# Load target NetCDF
yTarget = loadData.loadTarget(years = slice('1980-01-01', '2002-12-31'))
yTarget = yTarget.assign_coords(time = pd.to_datetime(yTarget.time.values).normalize())

# Initialize figure
fig, axes = plt.subplots(2, 2, figsize = (12, 6), squeeze = False)

# Custom colorbar (P02, mean and P98)
cmapMean = cm.get_cmap('RdBu_r', 16)
minMean = -32; maxMean = 32
boundsMean = [-32, -28, -24, -20, -16, -12, -8, -4, 0, 4, 8, 12, 16, 20, 24, 28, 32]
normMean = mpl.colors.BoundaryNorm(boundsMean, cmapMean.N)
ticksColorbarMean = [-32, -24, -16, -8, 0, 8, 16, 24, 32]

# Custom colorbar (sd)
cmapSD = cm.get_cmap('Purples', 10)
minSD = 0; maxSD = 20
boundsSD = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
normSD = mpl.colors.BoundaryNorm(boundsSD, cmapSD.N)
ticksColorbarSD = [0, 4, 8, 12, 16, 20]

# IPCC Regions
ar6 = regionmask.defined_regions.ar6.all
northAmerica = ar6[regions]

# Dict defining coords to print region mean values
dictCoord = {'NWN': [0.4, 0.85],
                'NEN': [0.76, 0.83],
                'WNA': [0.45, 0.55],
                'CNA': [0.65, 0.53],
                'ENA': [0.78, 0.45],
                'NCA': [0.585, 0.255],
                'SCA': [0.69, 0.1],
                'CAR': [0.89, 0.11]}

i, j = 0, 0

# P02 across time
yTargetP02 = da.apply_along_axis(utils.P02, 0, yTarget.tas)
yTargetP02 = yTargetP02.compute()
yTargetP02 = utils.maskRegions(yTargetP02, regionsToRemove)
yTargetP02 = utils.applyMaskNumpy(grid = yTargetP02)

axes[i, j] = northAmerica.plot(ax = axes[i, j],
                                add_ocean = False,
                                add_label = False,
                                line_kws = dict(linewidth = 1),
                                coastlines = False)

dictMeans = utils.computeRegionMeans(yTargetP02, regions)
for region in regions:
    axes[i, j].text(dictCoord[region][0],
                    dictCoord[region][1],
                    s = r'$\bf{' + str(dictMeans[region]) + '}$',
                    fontsize = 10,
                    horizontalalignment = 'center',
                    verticalalignment = 'center',
                    transform = axes[i, j].transAxes)

axes[i, j].set_title('P02 Temperature (C$^\circ$)', fontsize = 12)
axes[i, j].set_aspect('equal')

map = Basemap(ax = axes[i, j],
                llcrnrlon = -164.75, llcrnrlat = 11.75,
                urcrnrlon = -59.75, urcrnrlat = 69.75,
                resolution = 'c')
map.drawcoastlines(linewidth = 0.6)

im = map.imshow(yTargetP02, vmin = minMean, vmax = maxMean,
                cmap = cmapMean)

# Colorbar
divider = make_axes_locatable(axes[i, j])
cax = divider.append_axes('right', size = '5%', pad = 0.05)
fig.colorbar(im, cax = cax,
                orientation = 'vertical',
                norm = normMean,
                ticks = ticksColorbarMean)

i, j = 0, 1

# Mean across time
yTargetMean = da.apply_along_axis(utils.mean, 0, yTarget.tas)
yTargetMean = yTargetMean.compute()
yTargetMean = utils.maskRegions(yTargetMean, regionsToRemove)
yTargetMean = utils.applyMaskNumpy(grid = yTargetMean)

axes[i, j] = northAmerica.plot(ax = axes[i, j],
                                add_ocean = False,
                                add_label = False,
                                line_kws = dict(linewidth = 1),
                                coastlines = False)

dictMeans = utils.computeRegionMeans(yTargetMean, regions)
for region in regions:
    axes[i, j].text(dictCoord[region][0],
                    dictCoord[region][1],
                    s = r'$\bf{' + str(dictMeans[region]) + '}$',
                    fontsize = 10,
                    horizontalalignment = 'center',
                    verticalalignment = 'center',
                    transform = axes[i, j].transAxes)

axes[i, j].set_title('Mean Temperature (C$^\circ$)', fontsize = 12)
axes[i, j].set_aspect('equal')

map = Basemap(ax = axes[i, j],
                llcrnrlon = -164.75, llcrnrlat = 11.75,
                urcrnrlon = -59.75, urcrnrlat = 69.75,
                resolution = 'c')
map.drawcoastlines(linewidth = 0.6)

im = map.imshow(yTargetMean, vmin = minMean, vmax = maxMean,
                cmap = cmapMean)

# Colorbar
divider = make_axes_locatable(axes[i, j])
cax = divider.append_axes('right', size = '5%', pad = 0.05)
fig.colorbar(im, cax = cax,
                orientation = 'vertical',
                norm = normMean,
                ticks = ticksColorbarMean)

i, j = 1, 0

# P98 across time
yTargetP98 = da.apply_along_axis(utils.P98, 0, yTarget.tas)
yTargetP98 = yTargetP98.compute()
yTargetP98 = utils.maskRegions(yTargetP98, regionsToRemove)
yTargetP98 = utils.applyMaskNumpy(grid = yTargetP98)

axes[i, j] = northAmerica.plot(ax = axes[i, j],
                                add_ocean = False,
                                add_label = False,
                                line_kws = dict(linewidth = 1),
                                coastlines = False)

dictMeans = utils.computeRegionMeans(yTargetP98, regions)
for region in regions:
    axes[i, j].text(dictCoord[region][0],
                    dictCoord[region][1],
                    s = r'$\bf{' + str(dictMeans[region]) + '}$',
                    fontsize = 10,
                    horizontalalignment = 'center',
                    verticalalignment = 'center',
                    transform = axes[i, j].transAxes)

axes[i, j].set_title('P98 Temperature (C$^\circ$)', fontsize = 12)
axes[i, j].set_aspect('equal')

map = Basemap(ax = axes[i, j],
                llcrnrlon = -164.75, llcrnrlat = 11.75,
                urcrnrlon = -59.75, urcrnrlat = 69.75,
                resolution = 'c')
map.drawcoastlines(linewidth = 0.6)

im = map.imshow(yTargetP98, vmin = minMean, vmax = maxMean,
                cmap = cmapMean)

# Colorbar
divider = make_axes_locatable(axes[i, j])
cax = divider.append_axes('right', size = '5%', pad = 0.05)
fig.colorbar(im, cax = cax,
                orientation = 'vertical',
                norm = normMean,
                ticks = ticksColorbarMean)

i, j = 1, 1

# Std across time
yTargetSD = da.apply_along_axis(utils.standDev, 0, yTarget.tas)
yTargetSD = yTargetSD.compute()
yTargetSD = utils.maskRegions(yTargetSD, regionsToRemove)
yTargetSD = utils.applyMaskNumpy(grid = yTargetSD)

axes[i, j] = northAmerica.plot(ax = axes[i, j],
                                add_ocean = False,
                                add_label = False,
                                line_kws = dict(linewidth = 1),
                                coastlines = False)

dictMeans = utils.computeRegionMeans(yTargetSD, regions)
for region in regions:
    axes[i, j].text(dictCoord[region][0],
                    dictCoord[region][1],
                    s = r'$\bf{' + str(dictMeans[region]) + '}$',
                    fontsize = 10,
                    horizontalalignment = 'center',
                    verticalalignment = 'center',
                    transform = axes[i, j].transAxes)

axes[i, j].set_title('Standard Deviation of Temperature', fontsize = 12)
axes[i, j].set_aspect('equal')

map = Basemap(ax = axes[i, j],
                llcrnrlon = -164.75, llcrnrlat = 11.75,
                urcrnrlon = -59.75, urcrnrlat = 69.75,
                resolution = 'c')
map.drawcoastlines(linewidth = 0.6)

im = map.imshow(yTargetSD, vmin = minSD, vmax = maxSD,
                cmap = cmapSD)

# Colorbar
divider = make_axes_locatable(axes[i, j])
cax = divider.append_axes('right', size = '5%', pad = 0.05)
fig.colorbar(im, cax = cax,
                orientation = 'vertical',
                norm = normSD,
                ticks = ticksColorbarSD)

# Save final plot
plt.savefig(FIGS_PATH + 'fig-Climatology.pdf',
            dpi = 300, bbox_inches = 'tight',
            pad_inches = 0)
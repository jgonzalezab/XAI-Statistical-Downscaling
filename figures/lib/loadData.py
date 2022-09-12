import glob
import xarray as xr

# Target (Daymet025, Daymet05, EWEMBI05)
def loadTarget(years, chunks = None):

    DATA_PATH = '../preprocessData/data/'

    y = xr.open_dataset(DATA_PATH + 'y.nc4', chunks = chunks)
    y = y.sel(time = years)

    return y

# Prediction over train set
def loadTrainPred(model, chunks = None):

    DATA_PATH = '../statistical-downscaling/data/'

    yPred = xr.open_dataset(DATA_PATH + 'predsTrain_' + model + '.nc4',
                            chunks = chunks)

    return yPred

# Prediction over test set
def loadTestPred(model, chunks = None):

    DATA_PATH = '../statistical-downscaling/data/'

    yPred = xr.open_dataset(DATA_PATH + 'predsTest_' + model + '.nc4',
                            chunks = chunks)

    return yPred

# Prediction over historical GCM
def loadHistoricalPredProj(model, years ,
                           chunks = None):

    DATA_PATH = '../statistical-downscaling/data/'

    yProj = xr.open_dataset(DATA_PATH + 'predsGCM_Hist_' + model + '.nc4',
                            chunks = chunks)
    yProj = yProj.sel(time = years)

    return yProj

# Prediction over future GCM
def loadFuturePredProj(model, years, chunks = None):

    DATA_PATH = '../statistical-downscaling/data/'

    if years == slice('2006-01-01', '2040-12-31'):
        yProj = xr.open_dataset(DATA_PATH + 'predsGCM_Fut_2006_2040_' + model + '.nc4',
                            chunks = chunks)
    elif years == slice('2041-01-01', '2070-12-31'):
        yProj = xr.open_dataset(DATA_PATH + 'predsGCM_Fut_2041_2070_' + model + '.nc4',
                            chunks = chunks)
    elif years == slice('2071-01-01', '2100-12-31'):
        yProj = xr.open_dataset(DATA_PATH + 'predsGCM_Fut_2071_2100_' + model + '.nc4',
                            chunks = chunks)
    else:
        raise ValueError('Please provide a valid years slice:\n'\
                         'slice("2006-01-01", "2040-12-31")\n'\
                         'slice("2041-01-01", "2070-12-31")\n'\
                         'slice("2071-01-01", "2100-12-31")')

    return yProj

# Projection of GCM on historical period
def loadHistProj(years, chunks = None):

    DATA_PATH = '../preprocessData/data/'

    yProj = xr.open_dataset(DATA_PATH + 'y_GCM_historical.nc4',
                            chunks = chunks)
    yProj = yProj.sel(time = years)
    
    return yProj

# Projection of GCM on future period
def loadFutureProj(years,
                   chunks = None):

    DATA_PATH = '../preprocessData/data/'

    if years == slice('2006-01-01', '2040-12-31'):
        yProj = xr.open_dataset(DATA_PATH + 'y_GCM_future_2006_2040.nc4',
                                chunks = chunks)
    elif years == slice('2041-01-01', '2070-12-31'):
        yProj = xr.open_dataset(DATA_PATH + 'y_GCM_future_2041_2070.nc4',
                                chunks = chunks)
    elif years == slice('2071-01-01', '2100-12-31'):
        yProj = xr.open_dataset(DATA_PATH + 'y_GCM_future_2071_2100.nc4',
                                chunks = chunks)
    else:
        raise ValueError('Please provide a valid years slice:\n'\
                         'slice("2006-01-01", "2040-12-31")\n'\
                         'slice("2041-01-01", "2070-12-31")\n'\
                         'slice("2071-01-01", "2100-12-31")')

    return yProj

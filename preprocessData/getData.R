# Load libraries
library(loadeR)
source('./lib/downloadData.R')

# PATHS
DATA_PATH <- './data/'

# Log into UDG
loginUDG('***', '***')

# Longitude and latitude values
lon <- c(-165, -60)
lat <- c(12, 70)

# Download and save reanalysis data (predictor)
dataset <- 'ECMWF_ERA-Interim-ESD'

vars <- c('z@500','z@700','z@850','z@1000', 
          'hus@500','hus@700','hus@850','hus@1000',
          'ta@500','ta@700','ta@850','ta@1000',
          'ua@500','ua@700','ua@850','ua@1000',
          'va@500','va@700','va@850','va@1000')

years <- 1979:2008

download_X(dataset=dataset, vars=vars, lon=lon, lat=lat, years=years)

# Download and save observation data (predictand)
dataset <- 'PIK_Obs-EWEMBI'
vars <- c('tas')
years <- 1979:2008

download_Y(dataset=dataset, vars=vars, lon=lon, lat=lat, years=years)

# Download and save GCM predictors
# It also interpolates GCM predictors to reanalysis resolution

load(paste0(DATA_PATH, 'x.rda'))

# Historical predictors
dataset <- 'CMIP5-subset_EC-EARTH_r12i1p1_historical'

vars <- c('z@500','z@700','z@850','z@1000', 
          'hus@500','hus@700','hus@850','hus@1000',
          'ta@500','ta@700','ta@850','ta@1000',
          'ua@500','ua@700','ua@850','ua@1000',
          'va@500','va@700','va@850','va@1000')

years <- 1979:2005

download_GCM_predictors_historical(dataset=dataset, xRef=x, vars=vars, lon=lon, lat=lat, years=years)

# Future predictors
dataset <- 'CMIP5-subset_EC-EARTH_r12i1p1_rcp85'

vars <- c('z@500','z@700','z@850','z@1000', 
          'hus@500','hus@700','hus@850','hus@1000',
          'ta@500','ta@700','ta@850','ta@1000',
          'ua@500','ua@700','ua@850','ua@1000',
          'va@500','va@700','va@850','va@1000')

years <- 2006:2100

download_GCM_predictors_future(dataset=dataset, xRef=x, vars=vars, lon=lon, lat=lat, years=years)

# Download and save GCM proyections
# It also interpolates GCM proyections to observation resolution

load(paste0(DATA_PATH, 'y.rda'))

# Historical proyections
dataset <- 'CMIP5-subset_EC-EARTH_r12i1p1_historical'

vars <- c('tas')

years <- 1979:2005

download_GCM_proyections_historical(dataset=dataset, yRef=y, vars=vars, lon=lon, lat=lat, years=years)

# Future proyections
dataset <- 'CMIP5-subset_EC-EARTH_r12i1p1_rcp85'

vars <- c('tas')

years <- 2006:2100

download_GCM_proyections_future(dataset=dataset, yRef=y, vars=vars, lon=lon, lat=lat, years=years)
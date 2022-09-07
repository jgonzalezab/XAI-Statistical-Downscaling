# Load libraries
library(loadeR)
library(transformeR)
library(downscaleR)
library(visualizeR)
library(climate4R.value)
library(magrittr)
library(gridExtra)
library(RColorBrewer)
library(sp)
library(keras)
library(tensorflow)
library(downscaleR.keras)
library(loadeR.2nc)
library(reticulate)

download_X <- function(dataset, vars, lon, lat, years) {

  x <- lapply(vars, function(var) {
     print(paste0('Downloading variable ', var, ' of ', dataset, ' dataset'))
 	   loadGridData(dataset = dataset,
 	                var = var,
 	                lonLim = lon,
 	                latLim = lat,
 	                years = years)
 	 }) %>% makeMultiGrid()

  print(paste0('Saving as rda in ', DATA_PATH))
 	save(x, file = paste0(DATA_PATH, 'x.rda'))

}

download_Y <- function(dataset, vars, lon, lat, years) {

  y <- lapply(vars, function(var) {
     print(paste0('Downloading variable ', var, ' of ', dataset, ' dataset'))
 	   loadGridData(dataset = dataset,
 	                var = var,
 	                lonLim = lon,
 	                latLim = lat,
 	                years = years)
 	 }) %>% makeMultiGrid()

  print(paste0('Saving as rda in ', DATA_PATH))
 	save(y, file = paste0(DATA_PATH, 'y.rda'))

}

download_GCM_predictors_historical <- function(dataset, xRef, vars, lon, lat, years) {

  x_GCM <- lapply(vars, function(var) {
         print(paste0('Downloading variable ', var, ' of ', dataset, ' dataset'))
 	   		 loadGridData(dataset = dataset,
 	                	 	var = var,
 	                	  lonLim = lon,
 	                	 	latLim = lat,
 	                	 	years = years) %>%
 	   						interpGrid(new.coordinates = getGrid(xRef), method = 'bilinear')
 	 			 }) %>% makeMultiGrid()

  print(paste0('Saving as rds in ', DATA_PATH))
 	saveRDS(x_GCM, file = paste0(DATA_PATH, 'x_GCM_historical.rda'))

}

download_GCM_predictors_future <- function(dataset, xRef, vars, lon, lat, years) {

  x_GCM <- lapply(vars, function(var) {
         print(paste0('Downloading variable ', var, ' of ', dataset, ' dataset'))
 	   		 loadGridData(dataset = dataset,
 	                	 	var = var,
 	                	  lonLim = lon,
 	                	 	latLim = lat,
 	                	 	years = years) %>%
 	   						interpGrid(new.coordinates = getGrid(xRef), method = 'bilinear')
 	 			 }) %>% makeMultiGrid()

  print(paste0('Saving as rds in ', DATA_PATH))
 	saveRDS(x_GCM, file = paste0(DATA_PATH, 'x_GCM_future.rda'))

}

download_GCM_proyections_historical <- function(dataset, yRef, vars, lon, lat, years) {

  y_GCM <- lapply(vars, function(var) {
         print(paste0('Downloading variable ', var, ' of ', dataset, ' dataset'))
 	   		 loadGridData(dataset = dataset,
 	                	 	var = var,
 	                	  lonLim = lon,
 	                	 	latLim = lat,
 	                	 	years = years) %>%
 	   						interpGrid(new.coordinates = getGrid(yRef), method = 'bilinear')
 	 			 })

  print(paste0('Saving as rds in ', DATA_PATH))
 	saveRDS(y_GCM, file = paste0(DATA_PATH, 'y_GCM_historical.rda'))

}

download_GCM_proyections_future <- function(dataset, yRef, vars, lon, lat, years) {

  y_GCM <- lapply(vars, function(var) {
         print(paste0('Downloading variable ', var, ' of ', dataset, ' dataset'))
 	   		 loadGridData(dataset = dataset,
 	                	 	var = var,
 	                	  lonLim = lon,
 	                	 	latLim = lat,
 	                	 	years = years) %>%
 	   						interpGrid(new.coordinates = getGrid(yRef), method = 'bilinear')
 	 			 })

  print(paste0('Saving as rds in ', DATA_PATH))
 	saveRDS(y_GCM, file = paste0(DATA_PATH, 'y_GCM_future.rda'))

}

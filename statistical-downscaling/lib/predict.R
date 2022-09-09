deStandardize <- function(grid, meanGrid, stdGrid) {

    for (i in 1:dim(grid$Data)[1]) {
      	grid$Data[i, , ] <- (grid$Data[i, ,] * stdGrid) + meanGrid
   	}

    return(grid) 

}

scalingDeltaMapping <- function(grid, base, ref) {

    # Remove the seasonal trend
    grid_detrended <- scaleGrid(grid,
                                base = grid,
                                ref = base,
                                type = "center",
                                spatial.frame = "gridbox",
                                time.frame = "monthly")

    # Bias correct the mean and variance
    grid_detrended_corrected <- scaleGrid(grid_detrended,
                                          base = base,
                                          ref = ref,
                                          type = "standardize",
                                          spatial.frame = "gridbox",
                                          time.frame = "monthly")
    
    # Add the seasonal trend
    grid_corrected <- scaleGrid(grid_detrended_corrected,
                                base = base,
                                ref = grid,
                                type = "center",
                                spatial.frame = "gridbox",
                                time.frame = "monthly")

    return(grid_corrected)
}

predictTrain <- function(xTrain, yTrain, modelName, predictandStand) {

    # Filter days with NaNs or NA
    yTrain <- filterNA(yTrain)
    xTrain <- intersectGrid(xTrain, yTrain, which.return = 1)

    # Apply land mask or substitute NANs by zero if training CNN_UNET
    if (modelName == 'CNN_UNET') {
        yTrain$Data[is.na(yTrain$Data)] <- 0
    } else {
        yTrain <- applyMask(yTrain, yMask)
    }

    # Standardize predictand
    if(predictandStand) {

        yMean <- apply(yTrain$Data, c(2, 3), mean)
      	yStd <- apply(yTrain$Data, c(2, 3), sd)

        standString <- '_stand'

    } else {
        standString <- ''
    }

    # Standardize predictors
    xTrain_stand <- scaleGrid(xTrain, type = 'standardize')

    # Load xyT
    load(file = paste0(DATA_PATH_SD, 'xyT_', modelName, standString, '.rda'))

    # Prepare xyT
	xyT_train <- prepareNewData.keras(xTrain_stand, xyT)

    # Load model
    model <- load_model(model = modelName,
                        input_shape = dim(xyT$x.global)[-1],
                        output_shape = dim(xyT$y$Data)[2])

    model <- load_model_weights_hdf5(object = model,
									 filepath = paste0(MODELS_PATH, modelName, standString, '.h5'))

    # Compute predictions
    predsTrain <- downscalePredict.keras(xyT_train,
                               	 	     model = model,
                               	 	     loss = 'mse',
                               	 	     C4R.template = yTrain,
                               	 	     clear.session = TRUE)

    # Destandardize the predictand
    if(predictandStand) {
        predsTrain <- deStandardize(grid=predsTrain,
                                    meanGrid=yMean, stdGrid=yStd)
    }

    # Save the prediction as netCDF
	grid2nc(predsTrain, NetCDFOutFile = paste0(DATA_PATH_SD, 'predsTrain_',
                                               modelName, standString, '.nc4'))

}

predictTest <- function(xTrain, yTrain, xTest, yTest,
                        modelName, predictandStand) {

    # Filter days with NaNs or NA
    yTrain <- filterNA(yTrain)
    xTrain <- intersectGrid(xTrain, yTrain, which.return = 1)

    # Apply land mask or substitute NANs by zero if training CNN_UNET
    if (modelName == 'CNN_UNET') {
        yTrain$Data[is.na(yTrain$Data)] <- 0
    } else {
        yTrain <- applyMask(yTrain, yMask)
    }

    # Standardize predictand
    if(predictandStand) {

        yMean <- apply(yTrain$Data, c(2, 3), mean)
      	yStd <- apply(yTrain$Data, c(2, 3), sd)

        standString <- '_stand'

    } else {
        standString <- ''
    }

    # Standardize predictors
    xTest_stand <- scaleGrid(xTest, xTrain, type = 'standardize')

    # Load xyT
    load(file = paste0(DATA_PATH_SD, 'xyT_', modelName, standString, '.rda'))

    # Prepare xyT
	xyT_test <- prepareNewData.keras(xTest_stand, xyT)

    # Load model
    model <- load_model(model = modelName,
                        input_shape = dim(xyT$x.global)[-1],
                        output_shape = dim(xyT$y$Data)[2])
                        
    model <- load_model_weights_hdf5(object = model,
									 filepath = paste0(MODELS_PATH, modelName, standString, '.h5'))

    # Compute predictions
    predsTest <- downscalePredict.keras(xyT_test,
                               	 	    model = model,
                               	 	    loss = 'mse',
                               	 	    C4R.template = yTrain,
                               	 	    clear.session = TRUE)

    # Destandardize the predictand
    if(predictandStand) {
        predsTest <- deStandardize(grid=predsTest,
                                   meanGrid=yMean, stdGrid=yStd)
    }

    # Save the prediction as netCDF
	grid2nc(predsTest, NetCDFOutFile = paste0(DATA_PATH_SD, 'predsTest_',
                                              modelName, standString, '.nc4'))

}

predictGCM_Hist <- function(xTrain, yTrain, modelName, predictandStand) {

    # Filter days with NaNs or NA
    yTrain <- filterNA(yTrain)
    xTrain <- intersectGrid(xTrain, yTrain, which.return = 1)

    # Apply land mask or substitute NANs by zero if training CNN_UNET
    if (modelName == 'CNN_UNET') {
        yTrain$Data[is.na(yTrain$Data)] <- 0
    } else {
        yTrain <- applyMask(yTrain, yMask)
    }

    # Standardize predictand
    if(predictandStand) {

        yMean <- apply(yTrain$Data, c(2, 3), mean)
      	yStd <- apply(yTrain$Data, c(2, 3), sd)

        standString <- '_stand'

    } else {
        standString <- ''
    }

    # Load xyT
    load(file = paste0(DATA_PATH_SD, 'xyT_', modelName, standString, '.rda'))

    # Load GCM historical data
    x_GCM_Hist <- readRDS(paste0(DATA_PATH, 'x_GCM_historical.rds'))

    # Bias correct the GCM predictors
    x_GCM_Hist_BC <- scalingDeltaMapping(grid=x_GCM_Hist, base=x_GCM_Hist, ref=xTrain)

    # Standardize
    x_GCM_Hist_BC_stand <- scaleGrid(x_GCM_Hist_BC, xTrain, type='standardize')

    # Prepare xyT
	xyT_GCM_Hist <- prepareNewData.keras(x_GCM_Hist_BC_stand, xyT)

    # Load model
    model <- load_model(model = modelName,
                        input_shape = dim(xyT$x.global)[-1],
                        output_shape = dim(xyT$y$Data)[2])
                        
    model <- load_model_weights_hdf5(object = model,
									 filepath = paste0(MODELS_PATH, modelName, standString, '.h5'))

    # Compute predictions
    predsGCM_Hist <- downscalePredict.keras(xyT_GCM_Hist,
                               	 	        model = model,
                               	 	        loss = 'mse',
                               	 	        C4R.template = yTrain,
                               	 	        clear.session = TRUE)

    # Destandardize the predictand
    if(predictandStand) {
        predsGCM_Hist <- deStandardize(grid=predsGCM_Hist,
                                       meanGrid=yMean, stdGrid=yStd)
    }

    # Save the prediction as netCDF
	grid2nc(predsGCM_Hist, NetCDFOutFile = paste0(DATA_PATH_SD, 'predsGCM_Hist_',
                                                  modelName, standString, '.nc4'))

}

predictGCM_Fut <- function(xTrain, yTrain, modelName, predictandStand) {

    # Filter days with NaNs or NA
    yTrain <- filterNA(yTrain)
    xTrain <- intersectGrid(xTrain, yTrain, which.return = 1)

    # Apply land mask or substitute NANs by zero if training CNN_UNET
    if (modelName == 'CNN_UNET') {
        yTrain$Data[is.na(yTrain$Data)] <- 0
    } else {
        yTrain <- applyMask(yTrain, yMask)
    }

    # Standardize predictand
    if(predictandStand) {

        yMean <- apply(yTrain$Data, c(2, 3), mean)
      	yStd <- apply(yTrain$Data, c(2, 3), sd)

        standString <- '_stand'

    } else {
        standString <- ''
    }

    # Load xyT
    load(file = paste0(DATA_PATH_SD, 'xyT_', modelName, standString, '.rda'))

    # Load GCM historical data
    x_GCM_Hist <- readRDS(paste0(DATA_PATH, 'x_GCM_historical.rds'))

    # Load model
    model <- load_model(model = modelName,
                        input_shape = dim(xyT$x.global)[-1],
                        output_shape = dim(xyT$y$Data)[2])
                        
    model <- load_model_weights_hdf5(object = model,
									 filepath = paste0(MODELS_PATH, modelName, standString, '.h5'))

    # Periods
    yearsPeriods <- c('2006_2040', '2041_2070', '2071_2100')

    # Iterate over these periods
    for (period in yearsPeriods) {
        
        print(paste0('Computing GCM prediction on ', period))

        # Load GCM future data
        x_GCM_Fut <- readRDS(paste0(DATA_PATH, 'x_GCM_future_', period, '.rds'))

        # Bias correct the GCM predictors
        x_GCM_Fut_BC <- scalingDeltaMapping(grid=x_GCM_Fut, base=x_GCM_Hist, ref=xTrain)

        # Standardize
        x_GCM_Fut_BC_stand <- scaleGrid(x_GCM_Fut_BC, xTrain, type='standardize')

        # Prepare xyT
        xyT_GCM_Fut <- prepareNewData.keras(x_GCM_Fut_BC_stand, xyT)

        # Compute predictions
        predsGCM_Fut <- downscalePredict.keras(xyT_GCM_Fut,
                                               model = model,
                                               loss = 'mse',
                                               C4R.template = yTrain,
                                               clear.session = TRUE)

        # Destandardize the predictand
        if(predictandStand) {
            predsGCM_Fut <- deStandardize(grid=predsGCM_Fut,
                                          meanGrid=yMean, stdGrid=yStd)
        }

        # Save the prediction as netCDF
        grid2nc(predsGCM_Fut, NetCDFOutFile = paste0(DATA_PATH_SD, 'predsGCM_Fut_',
                                                      period, '_', modelName, standString, '.nc4'))

    }

}
# Mask observations (land gridpoints)
applyMask <- function(grid, mask) {

  for (i in 1:dim(grid$Data)[1]) {
    grid$Data[i, , ] <- grid$Data[i, , ] * mask
  }

  return(grid)

}

# Train and save a model
trainCNN <- function(xTrain, yTrain, yMask,
                     modelName,
                     connections,
                     predictandStand) {
    
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
        yTrain <- scaleGrid(yTrain, type = 'standardize', spatial.frame = 'gridbox') %>%
                            redim(drop = TRUE)

        standString <- '_stand'
    } else {
        standString <- ''
    }

    # Standardize predictors
    xTrain_stand <- scaleGrid(xTrain, type = 'standardize')

    # Prepare data for the model
    xyT <- prepareData.keras(x = xTrain_stand,
                             y = yTrain,
                             first.connection = connections[1],
                             last.connection = connections[2],
                             channels = 'last')

    # Load the model to train
    model <- load_model(model = modelName,
                        input_shape = dim(xyT$x.global)[-1],
                        output_shape = dim(xyT$y$Data)[2])

    print('****************************************')
    print(paste0('Training model ', modelName))
    print(paste0('Number of parameters: ', count_params(model)))
    print('****************************************')

    # Train and save the best model
    downscaleTrain.keras(obj = xyT,
                         model = model,
                         clear.session = TRUE,
                         compile.args = list('loss' = 'mse',
                                             'optimizer' = optimizer_adam(lr = 0.0001)),
                         fit.args = list('batch_size' = 100,
                                         'epochs' = 10000, 
                                         'validation_split' = 0.1,
                                         'verbose' = 1,
                                         'callbacks' = list(callback_early_stopping(patience = 30), 
                                                            callback_model_checkpoint(
                                                            filepath = paste0(MODELS_PATH, modelName, standString, '.h5'),
                                                            monitor = 'val_loss', save_best_only = TRUE,
                                                            save_weights_only = TRUE))))

    # Save xyT object to compute predictions
    save(xyT, file = paste0(DATA_PATH_SD, 'xyT_', modelName, standString, '.rda'))

}

# Train and save a model for epochs 800 and 3600 (overfitted training)
trainCNN_OF <- function(xTrain, yTrain, yMask,
                        modelName,
                        connections,
                        predictandStand) {
    
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
        yTrain <- scaleGrid(yTrain, type = 'standardize', spatial.frame = 'gridbox') %>%
                            redim(drop = TRUE)

        standString <- '_stand'
    } else {
        standString <- ''
    }

    # Standardize predictors
    xTrain_stand <- scaleGrid(xTrain, type = 'standardize')

    # Prepare data for the model
    xyT <- prepareData.keras(x = xTrain_stand,
                             y = yTrain,
                             first.connection = connections[1],
                             last.connection = connections[2],
                             channels = 'last')

    # Load the model to train
    model <- load_model(model = modelName,
                        input_shape = dim(xyT$x.global)[-1],
                        output_shape = dim(xyT$y$Data)[2])

    print('****************************************')
    print(paste0('Training model ', modelName))
    print(paste0('Number of parameters: ', count_params(model)))
    print('****************************************')

    # Implement a callback to save the model at specific epochs (800 and 3600)
    SpecificEpochSaving_Callback(keras$callbacks$Callback) %py_class% {
        on_epoch_end <- function(epoch, logs = NULL) {
            if ((epoch == 800) | (epoch == 3600)) {
                save_model_weights_hdf5(model,
                                        filepath = paste0(MODELS_PATH, modelName, standString, '_OF', epoch, '.h5'))
            }
        }       
    }

    # Train and save the best model
    downscaleTrain.keras(obj = xyT,
                         model = model,
                         clear.session = TRUE,
                         compile.args = list('loss' = 'mse',
                                             'optimizer' = optimizer_adam(lr = 0.0001)),
                         fit.args = list('batch_size' = 100,
                                         'epochs' = 5000, 
                                         'validation_split' = 0.1,
                                         'verbose' = 1,
                                         'callbacks' = list(callback_csv_logger(filename =
                                                                paste0(DATA_PATH_SD, 'trainLog_', modelName, standString, '.csv'),
                                                                separator=',', append=FALSE), 
                                                            SpecificEpochSaving_Callback())))

}
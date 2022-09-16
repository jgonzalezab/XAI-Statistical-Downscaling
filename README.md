# Using Explainability to Inform Deep Statistical Climate Downscaling Beyond Standard Validation Approaches

This repository contains the code to reproduce the paper *Using Explainability to Inform Deep Statistical Climate Downscaling Beyond Standard Validation Approaches* published in some Journal.

### Installation
To properly reproduce the environment required to run the experiments we rely on Docker. We provide two Dockerfiles with all the required libraries to train the models ([]()) and compute the saliency maps ([]()).

### Download and preprocess data
All the code for downloading and preprocessing the data is available in [preprocessData](). Note that downloading the data may require an account at [UDG-TAP](http://meteo.unican.es/udg-tap/home), however data can also be downloaded from the URLs of *Section 2.1 Region of study and data* of the paper.

All data can be downloaded by running [getData.R](). This task is time-consuming and requires about 16GB of disk space. While the data is being downloaded, the interpolation required by some of these datasets is also performed.

In the folder [runCluster]() we provide two `.sh` scripts to run this part of the paper following the workflow developed in [*A Container-Based Workflow for Distributed Training of Deep Learning Algorithms in HPC Clusters*](https://arxiv.org/abs/2208.02498). This workflow help us to run experiments in HPC clusters using Docker when GPUs are involved.

### Train the models and compute predictions
In [statistical-downscaling]() we provide the code to train the different deep learning models and compute the corresponding predictions for both reanalysis and GCM data. The [computeModel.R]() script calls the functions to perform this operations. By tuning some of its variables we control what model to run and whether to standardize the predictand. For example the CNNPan model requires the following values:

```
# Whether to standardize the predictand
predictandStand <- FALSE

# Train and save a model on the data
modelName <- 'CNNPan'
connections <- c('conv', 'dense')
```

Whereas CNN-UNET:
```
# Whether to standardize the predictand
predictandStand <- FALSE

# Train and save a model on the data
modelName <- 'CNN_UNET'
connections <- c('conv', 'conv')
```

The script [computeModel_OF.R]() train the models in the same way but overfitting them in order to save the models at epochs 800 and 3600 to reproduce the last part of the paper. We warn users that the training of CNN-UNET overfitted takes longer than the rest of the models (~2 days in one GPU).

For all these operations we rely on [climate4R](https://github.com/SantanderMetGroup/climate4R). We also provide the runCluster folder.

### Compute saliency maps
In the [XAI]() folder the code to compute the saliency maps for the different models in the training set can be found. For this part of the paper we use Python instead of R, since [INNvestigate](https://github.com/albermax/innvestigate) is only available for the former. Due to incompabilities issues between versions of different libraries, this environment is reproduced in a different Dockerfile []().

Similar to [computeModel.R](), the script [computeSM.py]() controls the computation of saliency maps for the different models and the different neurons (2000: north point, 5950: south point). For example, to compute the saliency maps of CNN_UNET at the north point we must execute the following command:

```
python computeSM.py CNN_UNET 2000
```

This saves the saliency maps as `.npy` in the [XAI/data/]() folder. As in the previous parts, we provide the runCluster folder.

### Generate figures
The [figures]() folder contains all the scripts to generate the figures shown in the paper. We rely on several python libraries like [xarray](https://github.com/pydata/xarray), [Dask](https://github.com/dask/dask) and [Matplotlib](https://github.com/matplotlib/matplotlib). Note that in order to successfully generate these figures it is necessary that all the previous phases have been executed correctly.
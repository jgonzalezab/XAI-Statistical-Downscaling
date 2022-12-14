# Using Explainability to Inform Deep Statistical Climate Downscaling Beyond Standard Validation Approaches

This repository contains the code to reproduce the paper *Using Explainability to Inform Deep Statistical Climate Downscaling Beyond Standard Validation Approaches*, published in Journal.

### Installation
To properly reproduce the environments necessary to run the experiments we rely on Docker. We provide two Dockerfiles with all the required libraries to download the data, train the models, compute the predictions ([Dockerfile_SD](https://github.com/jgonzalezab/XAI-Statistical-Downscaling/blob/main/docker/Dockerfile_SD)) and compute the saliency maps ([Dockerfile_XAI](https://github.com/jgonzalezab/XAI-Statistical-Downscaling/blob/main/docker/Dockerfile_XAI)). Inside the corresponding generated images we can find useful libraries on which we rely to run the experiments conforming this paper.

### Execution
The code in this repository is ordered following the different experiments performed in the paper:

1. Download and preprocess data ([preprocessData](https://github.com/jgonzalezab/XAI-Statistical-Downscaling/tree/main/preprocessData))
2. Train models and compute predictions ([statistical-downscaling](https://github.com/jgonzalezab/XAI-Statistical-Downscaling/tree/main/statistical-downscaling))
3. Compute saliency maps ([XAI](https://github.com/jgonzalezab/XAI-Statistical-Downscaling/tree/main/XAI))
4. Generate figures ([figures](https://github.com/jgonzalezab/XAI-Statistical-Downscaling/tree/main/figures))

Each of these folders contains its own README with instructions for execution

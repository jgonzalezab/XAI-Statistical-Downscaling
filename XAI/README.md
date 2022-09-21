### Compute saliency maps
Thhis folder contains the code to compute the saliency maps for the different models in the training set. For this part of the paper we use Python instead of R, since [INNvestigate](https://github.com/albermax/innvestigate) is only available for the former. Due to incompabilities issues between versions of different libraries, this environment must be reproduced with its own Dockerfile ([Dockerfile_XAI](https://github.com/jgonzalezab/XAI-Statistical-Downscaling/blob/main/docker/Dockerfile_XAI)).

Similar to `computeModel.R`, the script [computeSM.py](https://github.com/jgonzalezab/XAI-Statistical-Downscaling/blob/main/XAI/computeSM.py) controls the computation of saliency maps for the different models and neurons (2000: north point, 5950: south point). For example, to compute the saliency maps of CNN_UNET at the north point we must execute the following command:

```
python computeSM.py CNN_UNET 2000
```

This saves the saliency maps as `.npy` in the [XAI/data/](https://github.com/jgonzalezab/XAI-Statistical-Downscaling/tree/main/XAI/data) folder. As in previous parts, we provide the [runCluster](https://github.com/jgonzalezab/XAI-Statistical-Downscaling/tree/main/XAI/runCluster) folder.

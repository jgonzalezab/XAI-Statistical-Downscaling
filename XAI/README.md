### Compute saliency maps
The code to compute the saliency maps for the different models in the training set can be found. here For this part of the paper we use Python instead of R, since [INNvestigate](https://github.com/albermax/innvestigate) is only available for the former. Due to incompabilities issues between versions of different libraries, this environment ican be reproduced with its own Dockerfile []().

Similar to [computeModel.R](), the script [computeSM.py]() controls the computation of saliency maps for the different models and the different neurons (2000: north point, 5950: south point). For example, to compute the saliency maps of CNN_UNET at the north point we must execute the following command:

```
python computeSM.py CNN_UNET 2000
```

This saves the saliency maps as `.npy` in the [XAI/data/]() folder. As in previous parts, we provide the runCluster folder.
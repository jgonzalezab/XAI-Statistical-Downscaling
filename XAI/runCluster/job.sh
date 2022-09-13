#!/bin/bash

# Load some required modules
export PATH=$PATH:/bin/
source /etc/profile.d/modules.sh
module purge

udocker setup --nvidia --force $CONTAINER
nvidia-modprobe -u -c=0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/

# Models to compute saliency maps from
MODELS=('CNN10' 'CNN10_stand' 'CNNPan' 'CNN_UNET')

# Neurons from which saliency maps are computed
NEURONS=('2000' '5950')

# Run job
for model in "${MODELS[@]}"
do
  echo $model
  for neuron in "${NEURONS[@]}"
  do
      echo $neuron
      udocker run --hostenv --hostauth --user=$USER \
      -v $DIR_TO_MOUNT:/experiment/ $CONTAINER \
      /bin/bash -c "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/;
                    cd ./XAI/;
                    python computeSM.py ${model} ${neuron}"
   done
done
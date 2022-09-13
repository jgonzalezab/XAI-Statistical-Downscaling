#! /bin/bash

# Container name
export CONTAINER="deep-nm"

# Job config
export JOB_NAME="train_CNN_UNET"
export JOB2RUN="./job.sh"
export DIR_TO_MOUNT="/gpfs/projects/meteo/WORK/gonzabad/XAI-Statistical-Downscaling"

# Create slurm-log folder if it does not already exist
mkdir -p "${DIR_TO_MOUNT}/statistical-downscaling/cluster-log/"

# SLURM parameters
SLURM_PARAMS="--job-name=${JOB_NAME} --partition=wngpu --gres=gpu:2
              --time=72:00:00 --mem-per-cpu=128000 --output=${DIR_TO_MOUNT}/statistical-downscaling/cluster-log/${JOB_NAME}.out
              --error=${DIR_TO_MOUNT}/statistical-downscaling/cluster-log/${JOB_NAME}.err"
              
sbatch $SLURM_PARAMS $JOB2RUN
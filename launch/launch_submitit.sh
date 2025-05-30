#!/bin/bash

HOME=/home/user
PROJECT_DIR=$HOME/projects/reorder
CONFIG_PATH=$PROJECT_DIR/configs/final-exp-configs/longformer/longformer-in1k-snake-lr1e-4-bs576.yaml
GPU_TYPE=$1

RUN_NAME=$(python -c "from omegaconf import OmegaConf; print(OmegaConf.load('$CONFIG_PATH').get('run_name', 'unnamed'))")
EXP_NAME=$RUN_NAME

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate reorder

cd $PROJECT_DIR

PYTHONUNBUFFERED=1 \
python launch/submitit_train.py \
    --config $CONFIG_PATH \
    --gpu_type $GPU_TYPE \
    --nodes 1 \
    --partition standard \
    --account YOUR_ACCOUNT \
    --comment "$EXP_NAME on $GPU_TYPE" \
    --timeout 1440
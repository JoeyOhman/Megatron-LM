#!/bin/bash

PROJECT=/ceph/hpc/home/eujoeyo/group_space/joey/workspace/Megatron-LM
TARGET_DIR="/workspace/Megatron-LM"
CONTAINER_PATH="/ceph/hpc/home/eujoeyo/group_space/containers/megatron-deepspeed.sif"

singularity shell --nv --pwd /workspace/Megatron-LM --bind $PROJECT:$TARGET_DIR $CONTAINER_PATH

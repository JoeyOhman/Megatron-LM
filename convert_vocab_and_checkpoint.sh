#!/bin/bash

# MODEL_ID="base_600K"
MODEL_ID="large"
# ITERATION="0165000"
ITERATION=$1

WORKSPACE_PATH="/ceph/hpc/home/eujoeyo/group_space/joey/workspace"
# VOCAB_PATH="${WORKSPACE_PATH}/Megatron-LM/joey/data/vocab.txt"
CHECKPOINT_PATH="${WORKSPACE_PATH}/Megatron-LM/checkpoints/bert_$MODEL_ID/iter_$ITERATION/mp_rank_00/model_optim_rng.pt"

TRANSFORMER_UTILS_PATH="${WORKSPACE_PATH}/transformer-utils"

# SAVE_PRETRAINED_MODEL_PATH="./pretrained_model_hf_${MODEL_ID}"
SAVE_PRETRAINED_MODEL_PATH="./converted_to_hf_${MODEL_ID}/checkpoint_${ITERATION}"

cp ${TRANSFORMER_UTILS_PATH}/convert_megatron_bert_checkpoint.py ./convert_megatron_bert_checkpoint.py

# cmd_convert="python3 ${TRANSFORMER_UTILS_PATH}/convert_megatron_bert_checkpoint.py \
cmd_convert="python3 convert_megatron_bert_checkpoint.py \
              $CHECKPOINT_PATH \
              --save_path $SAVE_PRETRAINED_MODEL_PATH"

# cmd_tokenizer="python3 ${TRANSFORMER_UTILS_PATH}/vocab_to_tokenizer.py \
#               $VOCAB_PATH \
#               --save_path $SAVE_PRETRAINED_MODEL_PATH"

echo "$cmd_convert"
$cmd_convert

rm ./convert_megatron_bert_checkpoint.py

# echo "$cmd_tokenizer"
# $cmd_tokenizer

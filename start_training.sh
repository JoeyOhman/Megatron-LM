#!/bin/bash

CHECKPOINT_PATH=checkpoints/bert_base
VOCAB_FILE=joey/data/vocab.txt
DATA_PATH=joey/data/preprocessed/my-bert_text_sentence


DISTRIBUTED_ARGS="--nproc_per_node $NPROC_PER_NODE \
                  --nnodes $SLURM_JOB_NUM_NODES \
                  --node_rank $SLURM_NODEID \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# Intermediate size is set to: hidden-size * 4 = 3072
BERT_ARGS="--num-layers 12 \
           --hidden-size 768 \
           --num-attention-heads 12 \
           --seq-length 512 \
           --max-position-embeddings 512 \
           --lr 0.0001 \
           --lr-decay-iters 990000 \
           --train-iters 2000000 \
           --min-lr 0.00001 \
           --lr-warmup-fraction 0.01 \
	         --micro-batch-size 4 \
           --global-batch-size 8 \
           --vocab-file $VOCAB_FILE \
           --split 949,50,1 \
           --fp16"

OUTPUT_ARGS="--log-interval 10 \
             --save-interval 500 \
             --eval-interval 100 \
             --eval-iters 10 \
             --activations-checkpoint-method uniform"

# python -m pretrain_bert.py \
cmd="python3 -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_bert.py \
       $BERT_ARGS \
       $OUTPUT_ARGS \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH"

echo "Executing Command:"
echo $cmd

$cmd


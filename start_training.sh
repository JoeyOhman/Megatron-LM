#!/bin/bash

export OMP_NUM_THREADS=2

CHECKPOINT_PATH=checkpoints/bert_base
VOCAB_FILE=joey/data/robin-vocab.txt
# DATA_PATH=joey/data/preprocessed/my-bert_text_sentence
# DATA_PATH=joey-small-data_text_sentence
DATA_PATH_SHARDED=/ceph/hpc/home/eujoeyo/group_space/data/text/megatron_bert_data/sharded_preprocessed
DATA_NAMES=""
for i in {0..10}
  do
    DATA_NAMES="${DATA_NAMES} 1 ${DATA_PATH_SHARDED}/kb-data-shard_${i}_text_sentence"
  done

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
           --lr 7e-4 \
           --train-iters 600000 \
           --lr-warmup-iters 10000 \
	         --micro-batch-size 64 \
           --global-batch-size 2048 \
           --adam-beta2 0.999 \
           --adam-eps 1e-6 \
           --vocab-file $VOCAB_FILE \
           --split 949,50,1 \
           --fp16 \
           --tokenizer-type BertWordPieceCase"
# --lr 0.0001 \
# --lr-warmup-fraction 0.01 \
# --train-iters 2000000 \
# --lr-warmup-fraction 0.001 \
# --lr-decay-iters 990000 \
# --min-lr 0.00001 \
# roberta: --adam-beta2 0.98 \

OUTPUT_ARGS="--log-interval 100 \
             --save-interval 5000 \
             --eval-interval 1000 \
             --eval-iters 10 \
             --activations-checkpoint-method uniform"

# python -m pretrain_bert.py \
cmd="python3 -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_bert.py \
       $BERT_ARGS \
       $OUTPUT_ARGS \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_NAMES"

#

echo "Executing Command:"
echo $cmd

$cmd


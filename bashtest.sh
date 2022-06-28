#!/bin/bash

DATA_PATH_SHARDED=/ceph/hpc/home/eujoeyo/group_space/data/text/megatron_bert_data/sharded_preprocessed

DATA_NAMES=""
for i in {0..10}
  do
    DATA_NAMES="${DATA_NAMES} ${DATA_PATH_SHARDED}/kb-data-shard_${i}_text_sentence 1"
  done

echo $DATA_NAMES

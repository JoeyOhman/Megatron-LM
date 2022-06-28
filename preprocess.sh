#!/bin/bash
# data: /ceph/hpc/home/eujoeyo/group_space/data/text/all_text.docs, 74GB 1 row/doc raw text
# tokenizer: eurobink hface_bert mappen

MEGATRON_DATA_PATH="/ceph/hpc/home/eujoeyo/group_space/data/text/megatron_bert_data"
RAW_DATA_FILE="${MEGATRON_DATA_PATH}/all_text.docs"
OUT_DATA_FILE="${MEGATRON_DATA_PATH}/kbcorpus.json"
OUT_DATA_SHARD_PREFIX="${MEGATRON_DATA_PATH}/sharded/kbcorpus_shard"

VOCAB_PATH="/ceph/hpc/home/eujoeyo/group_space/joey/workspace/Megatron-LM/joey/data/robin-vocab.txt"

cmd_to_json="python joey/data/text_to_jsonl.py \
        --input_data_file ${RAW_DATA_FILE} \
        --output_data_file ${OUT_DATA_FILE}"

# echo "$cmd_to_json"
# $cmd_to_json

# OUT_DATA_SHARD_PREFIX="${MEGATRON_DATA_PATH}/sharded/shard7split/kbcorpus_shard"
# cmd_split_to_shards="python joey/data/split_into_shards.py\
#         --input_data_file ${MEGATRON_DATA_PATH}/sharded/kbcorpus_shard_7.json \
#         --output_data_file_prefix ${OUT_DATA_SHARD_PREFIX}"

# echo "$cmd_split_to_shards"
# $cmd_split_to_shards
# exit

# --input $OUT_DATA_FILE \
# --output-prefix kb-data \

# --input /ceph/hpc/home/eujoeyo/group_space/joey/workspace/Megatron-LM/joey/data/my-corpus.json \
# --output-prefix joey-small-data \

# --input $OUT_DATA_FILE \
for i in {0..10}
  do
    cmd_preprocess="python tools/preprocess_data.py \
           --input ${OUT_DATA_SHARD_PREFIX}_${i}.json \
           --output-prefix kb-data-shard_${i} \
           --vocab $VOCAB_PATH \
           --dataset-impl mmap \
           --tokenizer-type BertWordPieceCase \
           --split-sentences \
           --workers 100 \
           --log-interval 100000"
    echo "Executing command: ${cmd_preprocess}"
    $cmd_preprocess
  done

# --workers $SLURM_JOB_CPUS_PER_NODE"

# echo "$cmd_preprocess"
# sleep 3
# $cmd_preprocess

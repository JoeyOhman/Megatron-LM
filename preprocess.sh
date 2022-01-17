#!/bin/bash

python tools/preprocess_data.py \
       --input joey/data/my-corpus.json \
       --output-prefix my-bert \
       --vocab joey/data/vocab.txt \
       --dataset-impl mmap \
       --tokenizer-type BertWordPieceCase \
       --split-sentences


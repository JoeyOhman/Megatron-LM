#!/bin/bash

cd ..
python tools/preprocess_data.py \
       --input data/my-corpus.json \
       --output-prefix my-bert \
       --vocab data/vocab.txt \
       --dataset-impl mmap \
       --tokenizer-type BertWordPieceCase \
       --split-sentences


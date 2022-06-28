#!/bin/bash

for iter in {5000..165000..5000}
  do
    echo $iter
    iter_string="$iter"
    # prepend zeroes to match Megatron's iteration naming convention
    while [ ${#iter_string} -lt 7 ]
      do
        iter_string="0$iter_string"
        # echo "Prepended 0"
        # echo $iter_string
      done
    # exit
    ./convert_vocab_and_checkpoint.sh $iter_string
    # exit
  done

#!/bin/bash
python src/infer_mc.py \
    --model_name_or_path model_mc/ \
    --tokenizer_name model_mc/ \
    --test_file $2 \
    --context_file $1 \
    --test_output test_mc_out.json \
    --max_seq_length 512 \
    --per_device_eval_batch_size 64 \

python src/infer_qa.py \
    --model_name_or_path model_qa \
    --tokenizer_name model_qa \
    --test_file test_mc_out.json \
    --context_file $1 \
    --test_output $3 \
    --max_seq_length 512 \
    --doc_stride 128 \
    --per_device_eval_batch_size 64 \
    --preprocessing_num_workers 1 \

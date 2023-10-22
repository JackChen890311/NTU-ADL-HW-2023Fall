#!/bin/bash
python infer_mc.py \
    --model_name_or_path output/lert_mc/ \
    --tokenizer_name output/lert_mc/ \
    --test_file data/test.json \
    --context_file data/context.json \
    --test_output result/test_mc_out.json \
    --max_seq_length 512 \
    --per_device_eval_batch_size 64 \

python infer_qa.py \
    --model_name_or_path output/lert_qa/ \
    --tokenizer_name output/lert_qa/ \
    --test_file result/test_mc_out.json \
    --context_file data/context.json \
    --test_output result/qa_out.csv \
    --max_seq_length 512 \
    --doc_stride 128 \
    --per_device_eval_batch_size 64 \
    --preprocessing_num_workers 1 \
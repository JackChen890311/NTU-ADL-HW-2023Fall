#!/bin/bash
python train_mc.py \
    --model_type bert \
    --tokenizer_name bert-base-chinese \
    --train_file data/train.json \
    --validation_file data/valid.json \
    --context_file data/context.json \
    --max_seq_length 512 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 5 \
    --output_dir output/bert_mc_scratch \
    --with_tracking \
    # --debug \
    # --dataset_name swag

python train_qa.py \
    --model_type bert \
    --tokenizer_name bert-base-chinese \
    --train_file data/train.json \
    --validation_file data/valid.json \
    --context_file data/context.json \
    --max_seq_length 512 \
    --doc_stride 128 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --preprocessing_num_workers 1 \
    --num_train_epochs 5 \
    --output_dir output/bert_qa_scratch \
    --with_tracking \
    # --debug \
    # --dataset_name squad \

#     --model_type bert \
#     --tokenizer_name bert-base-chinese \
#     --num_train_epochs 10 \
    # --model_name_or_path hfl/chinese-lert-base \
    # --tokenizer_name hfl/chinese-lert-base \
    # --model_name_or_path hfl/chinese-lert-large \
    # --tokenizer_name hfl/chinese-lert-large \
#!/bin/bash
python3 train.py \
    --model_name_or_path google/mt5-small \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --preprocessing_num_workers 16 \
    --output_dir ./output/6/ \
    --output_file ./output/6/public_result.jsonl \
    --text_column maintext \
    --summary_column title \
    --train_file ./data/train.jsonl \
    --validation_file ./data/public.jsonl \
    --num_train_epochs 15 \
    --num_beams 3 \
    --learning_rate 3e-4 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --num_warmup_steps 500 \
    --with_tracking \
    # --dataset_name cnn_dailymail \
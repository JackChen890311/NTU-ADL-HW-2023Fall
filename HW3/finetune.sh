#!bin/bash
python3 train.py \
    --model_name_or_path Taiwan-LLM-7B-v2.0-chat \
    --output_dir ./output/4bits \
    --dataset data/train.json \
    --dataset_format self-defined \
    --bits 4 \
    --bf16 \
    --do_train \
    --max_steps 1000 \
    --save_steps 100 \
    --per_device_train_batch_size 8 \
    --learning_rate 3e-5 \
    --max_train_epochs 3 \
    --evaluation_strategy "steps" \
    --eval_steps "100" \
    --gradient_accumulation_steps 2 \
    --overwrite_output_dir \
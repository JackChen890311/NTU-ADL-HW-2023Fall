#!/bin/bash
python3 infer.py \
    --validation_file $1 \
    --output_file $2 \
    --source_prefix "summarize: " \
    --text_column maintext \
    --num_beams 3 \
    --per_device_eval_batch_size 16 \
    --model_name_or_path ./output/1/ \
    # remember to modify the model_name_or_path
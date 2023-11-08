#!/bin/bash
python3 infer.py \
    --validation_file $1 \
    --output_file $2 \
    --source_prefix "summarize: " \
    --text_column maintext \
    --num_beams 20 \
    --top_p 1 \
    --top_k 50 \
    --temperature 1 \
    --per_device_eval_batch_size 2 \
    --model_name_or_path ./model \
    # --do_sample \

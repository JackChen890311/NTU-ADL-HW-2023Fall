#!bin/bash
python3 infer.py \
    --base_model $1 \
    --peft_model $2 \
    --input_json $3 \
    --output_json $4 \

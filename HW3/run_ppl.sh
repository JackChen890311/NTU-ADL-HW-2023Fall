#!bin/bash
python3 ppl.py \
    --base_model_path Taiwan-LLM-7B-v2.0-chat \
    --peft_path output/checkpoint-10 \
    --test_data_path data/public_test.json \
python run_swag_no_trainer.py \
    --with_tracking \
    --train_file data/train.json \
    --validation_file data/valid.json \
    --context_file data/context.json \
    --model_name_or_path bert-base-chinese \
    --max_seq_length 512 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --output_dir output_swag \

# --debug
python run_mc_no_trainer.py \
    --model_name_or_path hfl/chinese-pert-base \
    --tokenizer_name hfl/chinese-pert-base \
    --train_file data/train.json \
    --validation_file data/valid.json \
    --context_file data/context.json \
    --max_seq_length 512 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --output_dir output/pert_mc \
    # --debug
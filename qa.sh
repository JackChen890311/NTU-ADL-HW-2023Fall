python run_qa_no_trainer.py \
    --model_name_or_path hfl/chinese-pert-large \
    --tokenizer_name hfl/chinese-pert-large \
    --train_file data/train.json \
    --validation_file data/valid.json \
    --context_file data/context.json \
    --max_seq_length 512 \
    --doc_stride 128 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --preprocessing_num_workers 8 \
    --output_dir output/pert_qa \
    # --debug \
    # --do_predict \
    # --test_file data/test.json \
    # --dataset_name squad \
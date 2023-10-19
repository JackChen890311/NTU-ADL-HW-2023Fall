python infer_mc.py \
    --model_name_or_path output/lert_mc/ \
    --tokenizer_name output/lert_mc/ \
    --test_file data/test.json \
    --context_file data/context.json \
    --max_seq_length 512 \
    --per_device_eval_batch_size 64 \

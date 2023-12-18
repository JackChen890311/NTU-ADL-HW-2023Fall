#!bin/bash
# for n in {100..1000..100}; 
# do  
#     echo $n
#     python3 ppl.py \
#         --base_model_path Taiwan-LLM-7B-v2.0-chat \
#         --peft_path output/4bits/checkpoint-$n \
#         --test_data_path data/public_test.json
# done

python3 ppl.py \
    --base_model_path Taiwan-LLM-7B-v2.0-chat \
    --peft_path adapter_checkpoint/ \
    --test_data_path data/public_test.json
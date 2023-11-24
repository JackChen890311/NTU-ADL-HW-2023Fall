import transformers
from datasets import load_dataset
from utils import get_prompt, get_bnb_config
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

bnb_config = get_bnb_config()
model_id = "Taiwan-LLM-7B-v2.0-chat"
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

text = "請解釋何謂人工智慧"
device = "cuda:0"

# prompt = get_prompt(text)
# inputs = tokenizer(prompt, return_tensors="pt").to(device)
# outputs = model.generate(**inputs, max_new_tokens=128)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=8, 
    lora_alpha=32, 
    target_modules=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"], 
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)
print(model)

def preprocess(examples):
    examples['instruction'] = [get_prompt(text) for text in examples['instruction']]
    new_examples = tokenizer(examples['instruction'], padding='max_length', max_length=256,truncation=True)
    new_examples['labels'] = tokenizer(examples['output'], padding='max_length', max_length=256,truncation=True)['input_ids']
    new_examples['labels'] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in new_examples['labels']]
    return new_examples

data_files = {
    "train": "data/train.json",
    "eval": "data/public_test.json",
    # "test": "data/private_test.json"
}
data = load_dataset('json', data_files=data_files)
print(data)
# print(tokenizer(data['train'][0]["instruction"]))
# print(tokenizer(data['train'][0]["output"]))

train_dataset = data['train'].map(preprocess, batched=True, num_proc=4)
eval_dataset = data['eval'].map(preprocess, batched=True, num_proc=4)
# print(train_dataset[0])
for k in train_dataset[0]:
    print(k, len(train_dataset[0][k]))
# raise KeyboardInterrupt

tokenizer.pad_token = tokenizer.eos_token

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_dataset,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=1,
        max_steps=10,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="model",
        optim="paged_adamw_8bit"
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()
trainer.save_state()
trainer.save_model('model')

model.save_pretrained('model2')

import torch
trainable_weights = {k: v for k, v in model.named_parameters() if v.requires_grad}
print(trainable_weights.keys())
print(trainable_weights.values())
torch.save(model, 'adapter_model.pt')
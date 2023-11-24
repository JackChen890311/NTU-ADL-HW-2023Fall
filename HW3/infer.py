import transformers
from datasets import load_dataset
from utils import get_prompt, get_bnb_config
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model, PeftModelForCausalLM
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

bnb_config = get_bnb_config()

base_model = "Taiwan-LLM-7B-v2.0-chat"
peft_model = 'model'

config = LoraConfig.from_pretrained(peft_model)
model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb_config, device_map='cuda:0')
# model.load_adapter(peft_model,'lora')
tokenizer = AutoTokenizer.from_pretrained(base_model)
# model = get_peft_model(model, config)
# model = PeftModel.from_pretrained(model, peft_model)

fuck = get_peft_model(model, config)
# fuck = PeftModelForCausalLM.from_pretrained(fuck, 'peft_model', is_trainable=False, config=config)
print(fuck)
for k,v in fuck.named_parameters():
    print(k, v.shape)

import torch
sd = torch.load('adapter_model.pt')
print(sd.keys())


new_sd = {}
for k in sd:
    newk = k.split('.weight')[0] + '.default.weight'
    print(newk)
    new_sd[newk] = sd[k]

# for k,v in fuck.named_parameters():
#     if k not in new_sd:
#         new_sd[k] = v

print(fuck.load_state_dict(new_sd, strict=False))

#
text = "請解釋何謂人工智慧"
device = "cuda:0"

# model = model.to(device)
prompt = get_prompt(text)
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
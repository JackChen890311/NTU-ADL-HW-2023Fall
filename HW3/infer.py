import json
import argparse
from tqdm import tqdm
from utils import get_prompt, get_bnb_config
from peft import LoraConfig
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model',type=str,default="Taiwan-LLM-7B-v2.0-chat")
    parser.add_argument('--peft_model',type=str,default="output/step1000_8bits_lr/checkpoint-1000")
    parser.add_argument('--input_json',type=str,default="data/public_test.json")
    parser.add_argument('--output_json',type=str,default="output/output.json")
    args = parser.parse_args()

    bnb_config = get_bnb_config()

    config = LoraConfig.from_pretrained(args.peft_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, quantization_config=bnb_config, device_map='cuda:0')
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = PeftModel.from_pretrained(model, args.peft_model)

    with open(args.input_json, "r") as f:
        data = json.load(f)

    result = []
    for i in tqdm(range(len(data))):
        result_dict = {}
        prompt = get_prompt(data[i]['instruction'])
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=128)
        result_dict['id'] = data[i]['id']
        result_dict['output'] = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]
        result.append(result_dict)

    # print(result)
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
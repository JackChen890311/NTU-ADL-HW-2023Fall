from transformers import BitsAndBytesConfig
import torch


def get_prompt(instruction: str) -> str:
    '''Format the instruction as a prompt for LLM.'''
    # return f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。USER: {instruction} ASSISTANT:"
    return f"你是一個精通現代中文與文言文的大師，以下是用戶和你之間的對話。你的目標是對用戶的問題提供有用、精確且簡潔的回答。USER: {instruction} ASSISTANT:"

def get_bnb_config() -> BitsAndBytesConfig:
    '''Get the BitsAndBytesConfig.'''
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.float32,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
    )
    return bnb_config

def get_few_shot_prompt(instruction, idx, data, shotcnt):
    if shotcnt == 0:
        return get_prompt(instruction)
    else:
        example = '以下為幾個翻譯的正確例子\n'
        for i in range(shotcnt):
            example += f'USER:{data[idx-i]["instruction"]} ASSISTANT:{data[idx-i]["output"]} \n'

        example += '\n以下為你需要翻譯的句子，請根據前面提供的正確結果進行翻譯\n'
        prompt = f"你是一個精通現代中文與文言文的大師，以下是用戶和你之間的對話。你的目標是對用戶的問題提供有用、精確且簡潔的回答。{example} USER:{instruction} ASSISTANT:"
        return prompt
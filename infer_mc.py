import json
import torch
import numpy as np
import pandas as pd
import evaluate
import accelerate
from tqdm import tqdm
from datasets import load_dataset
from itertools import chain
from utils_mc import parse_args, DataCollatorForMultipleChoice
from torch.utils.data import DataLoader

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import PaddingStrategy, check_min_version, send_example_telemetry


def inference():
    args = parse_args()

    data_files = {}
    if args.test_file is not None:
        data_files['test'] = args.test_file
    raw_datasets = load_dataset('json', data_files=data_files)
    
    if args.context_file is not None:
        with open(args.context_file, 'r') as f:
            context = json.load(f)

    # ===================================================

    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name or args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
    )
    model = AutoModelForMultipleChoice.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        trust_remote_code=args.trust_remote_code,
    )
    
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    padding = "max_length" if args.pad_to_max_length else False


    def preprocess_function(examples):
        question = [[question] * 4 for question in examples['question']]
        answer = [[context[i] for i in options] for options in examples['paragraphs']]
        # labels = [examples['paragraphs'][i].index(examples['relevant'][i]) for i in range(len(examples['id']))]
        labels = [0] * len(examples['question'])

        # Flatten out
        question = list(chain(*question))
        answer = list(chain(*answer))

        # Tokenize
        tokenized_examples = tokenizer(
            question,
            answer,
            max_length=args.max_seq_length,
            padding=padding,
            truncation=True,
        )
        # Un-flatten
        tokenized_inputs = {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    processed_datasets = raw_datasets.map(
        preprocess_function, batched=True, remove_columns=raw_datasets["test"].column_names
    )

    test_dataset = processed_datasets["test"]

    data_collator = default_data_collator if args.pad_to_max_length else DataCollatorForMultipleChoice(tokenizer, pad_to_multiple_of=None)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    
    # device = accelerate.device
    device = torch.device('cuda')
    model = model.to(device)

    # model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
    #     model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    # )

    model.eval()
    result = torch.zeros((0),dtype=int)
    for step, batch in tqdm(enumerate(test_dataloader)):
        for k in batch.keys():
            # print(k)
            # print(batch[k].shape)
            batch[k] = batch[k].to(device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        result = torch.cat((result, predictions.cpu()))
        # print(predictions)
        # break
    
    print(result.shape)
    df = pd.DataFrame({'Prediction':result.numpy()})
    df.to_csv('result/mc_out.csv')
    
    with open(args.test_file, 'r') as f:
        result_json = json.load(f)
    
    assert len(result_json) == result.shape[0]
    for idx in range(len(result)):
        result_json[idx]['relevant'] = int(result_json[idx]['paragraphs'][int(df.Prediction[idx])])
    
    with open('result/test_mc_out.json', 'w', encoding='utf-8') as f:
        json.dump(result_json, f, ensure_ascii=False)

if __name__ == "__main__":
    inference()

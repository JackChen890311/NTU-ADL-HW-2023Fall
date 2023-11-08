import nltk
import torch
import jsonlines
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
)

from utils import parse_args


def inference_all(args):
    data_files = {
        'validation': args.validation_file,
    }
    datasets = load_dataset('json', data_files=data_files)
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        trust_remote_code=args.trust_remote_code,
    )

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = args.source_prefix if args.source_prefix is not None else ""
    column_names = datasets["validation"].column_names
    text_column = args.text_column
    summary_column = args.text_column # not used
    id_column = 'id' # document id

    if args.val_max_target_length is None:
        args.val_max_target_length = args.max_target_length
    max_target_length = args.max_target_length
    padding = "max_length" if args.pad_to_max_length else False


    # ==================================================================================================
    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[summary_column]
        ids = examples[id_column]
        ids = [[int(i)] for i in ids]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        model_inputs["origin_id"] = ids
        return model_inputs
    # ==================================================================================================

    max_target_length = args.val_max_target_length
    eval_dataset = datasets["validation"].map(
        preprocess_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
    )

    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    inference_part(args, model, tokenizer, eval_dataloader)


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels
    

def inference_part(args, model, tokenizer, eval_dataloader):
    all_ids = []
    all_preds = []

    gen_kwargs = {
            "max_length": args.val_max_target_length,
            "num_beams": args.num_beams,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
        }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    model = model.to(device)

    for batch in tqdm(eval_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            generated_tokens = model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
            ).cpu()

            # Padding might be needed depending on your model
            if not args.pad_to_max_length:
                # If we did not pad to max length, we need to pad the labels too
                labels = batch["labels"].cpu()
                pad_token_id = tokenizer.pad_token_id
                while generated_tokens.shape[1] > labels.shape[1]:
                    labels = torch.cat([labels, torch.ones((labels.shape[0], 1), dtype=torch.long) * pad_token_id], dim=1)
                while labels.shape[1] > generated_tokens.shape[1]:
                    generated_tokens = torch.cat([generated_tokens, torch.ones((generated_tokens.shape[0], 1), dtype=torch.long) * pad_token_id], dim=1)

            generated_tokens = generated_tokens.cpu().numpy()
            labels = labels.cpu().numpy()

            if args.ignore_pad_token_for_loss:
                # Replace -100 in the labels as we can't decode them.
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
            original_ids = batch['origin_id']

            for id, pred in zip(original_ids, decoded_preds):
                all_ids.append(str(id.detach().cpu().item()))
                all_preds.append(pred)
                
    # with open(args.output_file, 'w') as f:
    #     for id, pred in zip(all_ids, all_preds):
    #         f.write(str({"title": pred, "id": id}).replace("'", '"')+'\n')

    with jsonlines.open(args.output_file, mode='w') as writer:
            for id, pred in zip(all_ids, all_preds):
                writer.write({"title": pred, "id": id})


if __name__ == '__main__':

    args = parse_args()
    print(args)

    inference_all(args)
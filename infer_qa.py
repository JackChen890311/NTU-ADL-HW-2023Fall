import json
import torch
import numpy as np
import pandas as pd
import evaluate
import accelerate
from datasets import load_dataset
from itertools import chain
from tqdm import tqdm
from utils_qa import parse_args, save_prefixed_metrics, postprocess_qa_predictions
from torch.utils.data import DataLoader

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
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
            CONTEXT = json.load(f)

    # ===================================================

    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name or args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            trust_remote_code=args.trust_remote_code,
    )
    
    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"
    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)




    # ============================================================================
    # Validation preprocessing
    def prepare_validation_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace

        myquestion = examples['question']
        mycontext = [CONTEXT[idx] for idx in examples['relevant']]
        # myanswer = []
        # for idx, a in enumerate(examples['answer']):
        #     myanswer.append({
        #         'text': [a['text']],
        #         'answer_start': [a['start']]
        #     })


        # examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            # examples[question_column_name if pad_on_right else context_column_name],
            # examples[context_column_name if pad_on_right else question_column_name],
            myquestion if pad_on_right else mycontext,
            mycontext if pad_on_right else myquestion,
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples
    

    # Post-processing:
    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=args.version_2_with_negative,
            n_best_size=args.n_best_size,
            max_answer_length=args.max_answer_length,
            null_score_diff_threshold=args.null_score_diff_threshold,
            output_dir=args.output_dir,
            prefix=stage,
            context=CONTEXT,
        )
        # Format the result to the format the metric expects.
        if args.version_2_with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        return formatted_predictions
        # print([ex[answer_column_name] for ex in examples])
        # ans = [{'answer_start': ex[answer_column_name]['start'], 'text':ex[answer_column_name]['text']} for ex in examples]
        # references = [{"id": ex["id"], "answers": ans} for ex in examples]
        # return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    # Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor
    def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
        """
        Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor

        Args:
            start_or_end_logits(:obj:`tensor`):
                This is the output predictions of the model. We can only enter either start or end logits.
            eval_dataset: Evaluation dataset
            max_len(:obj:`int`):
                The maximum length of the output tensor. ( See the model.eval() part for more details )
       """

        step = 0
        # create a numpy array and fill it with -100.
        logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
        # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather_for_metrics
        for i, output_logit in enumerate(start_or_end_logits):  # populate columns
            # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
            # And after every iteration we have to change the step

            batch_size = output_logit.shape[0]
            cols = output_logit.shape[1]

            if step + batch_size < len(dataset):
                logits_concat[step : step + batch_size, :cols] = output_logit
            else:
                logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

            step += batch_size

        return logits_concat
    
    # ============================================================================


    test_dataset = raw_datasets['test']
    test_dataset = test_dataset.map(
        prepare_validation_features,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=raw_datasets['test'].column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on prediction dataset",
    )


    data_collator = default_data_collator if args.pad_to_max_length else DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)
    # test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    test_dataset_for_model = test_dataset.remove_columns(["example_id", "offset_mapping"])
    test_dataloader = DataLoader(test_dataset_for_model, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    
    
    # device = accelerate.device
    device = torch.device('cuda')
    model = model.to(device)

    # model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
    #     model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    # )

    model.eval()
    all_start_logits = []
    all_end_logits = []

    result = torch.zeros((0),dtype=int)
    for step, batch in tqdm(enumerate(test_dataloader)):
        for k in batch.keys():
            # print(k)
            # print(batch[k].shape)
            batch[k] = batch[k].to(device)
        with torch.no_grad():
            outputs = model(**batch)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            all_start_logits.append(start_logits.cpu().numpy())
            all_end_logits.append(end_logits.cpu().numpy())
        # break

    max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor
    # concatenate the numpy array
    start_logits_concat = create_and_fill_np_array(all_start_logits, test_dataset, max_len)
    end_logits_concat = create_and_fill_np_array(all_end_logits, test_dataset, max_len)

    # delete the list of numpy arrays
    del all_start_logits
    del all_end_logits

    outputs_numpy = (start_logits_concat, end_logits_concat)
    prediction = post_processing_function(raw_datasets['test'], test_dataset, outputs_numpy)
    # print(prediction)

    predict_id = [prediction[idx]['id'] for idx in range(len(prediction))]
    predict_result = [prediction[idx]['prediction_text'] for idx in range(len(prediction))]

    df = pd.DataFrame({'id':predict_id, 'answer': predict_result})
    df.to_csv('result/qa_out.csv', index=False)

if __name__ == "__main__":
    inference()

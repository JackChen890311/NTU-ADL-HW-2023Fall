# NTU ADL HW1
R12922051 資工碩一 陳韋傑

## Code reference
 - For multuple choices: https://github.com/huggingface/transformers/blob/main/examples/pytorch/multiple-choice/run_swag_no_trainer.py
 - For extractive qa: https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_qa_no_trainer.py

## Environment setting
- Python = 3.9.18
- Use `pip install -r requirement.txt` to install packages

## Steps to run inference
- Inference code are in `infer_mc.py` and `infer_qa.py`
- Please run `bash download.sh` first to download dataset and trained models
- Then run `bash run.sh {path_to_context_json} {path_to_test_json} {path_to_output_csv}` and replace the paths as you wish
- The dataset should store in `/data`, so you should replace `{path_to_context_json}` and `{path_to_test_json}` to `data/context.json` and `data/context.json`
- When running inference, there will also be a generated file called `test_mc_out.json`, this file is the result from multiple choices and is used by extractive qa
- Personally I run `bash infer.sh` when I need to do inference, where the paths are hardcoded in the file
- If encountered **NotImplementedError: Loading a dataset cached in a LocalFileSystem is not supported.**, this is possibly due to `data/context.json` and `data/test.json` are cached in different environment, so please try to delete the cached file (I'm not sure why but fxck you hugging face)

## Steps to run trainning
- Trainning code are in `train_mc.py` and `train_qa.py`
- Run `bash train.sh` to train models, please change the `output_dir` before you run it, also make sure the data are stored in the correct directory
- Adjust the hyperparameters by yourself
- By default the two task will run sequentially, if you wish to run parallelly please modify the bash file and change the gpu device in `train_mc.py` and `train_qa.py` (line 60, `os.environ['CUDA_VISIBLE_DEVICES'] = '0'`) 
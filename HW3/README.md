# NTU ADL HW3
R12922051 資工碩一 陳韋傑

---

## Code reference
 - https://github.com/artidoro/qlora
 

## Environment setting
- Python = 3.10.13
- Use `pip install -r requirement.txt` to install packages
- Please also install bitsandbytes, 0.41.2 is preferred

## Steps to run inference
- Inference code are in `infer.py`
- Please run `bash download.sh` first to download trained models
- Then run `bash run.sh {base_model} {peft_model} {input_json} {output_json}` and replace the paths as you wish

## Steps to run trainning
- Trainning code are in `train.py`
- Run `bash train.sh` to train models, please change the `output_dir` before you run it, also make sure the data are stored in the correct directory
- Adjust the hyperparameters by yourself


## Dataset
[download link for dataset and Taiwan-LLaMa model](https://drive.google.com/drive/folders/1hyk6DjCQA9lMc0jGrqg7PMtjmrbs_fN5)


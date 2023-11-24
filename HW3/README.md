# NTU ADL HW3
R12922051 資工碩一 陳韋傑

---

## Code reference
 - https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization
 

## Environment setting
- Python = 3.9.18
- Use `pip install -r requirement.txt` to install packages
- Please also install matplotlib if plotting is needed

## Steps to run inference
- Inference code are in `infer.py`
- Please run `bash download.sh` first to download trained models
- Then run `bash run.sh {path_to_input_jsonl} {path_to_output_jsonl}` and replace the paths as you wish

## Steps to run trainning
- Trainning code are in `train.py`
- Run `bash train.sh` to train models, please change the `output_dir` and `output_file` before you run it, also make sure the data are stored in the correct directory
- Adjust the hyperparameters by yourself


## Dataset
[download link](https://drive.google.com/file/d/186ejZVADY16RBfVjzcMcz9bal9L3inXC/view?usp=sharing)

## Installation
```
git clone https://github.com/moooooser999/ADL23-HW2.git
cd ADL23-HW2
pip install -e tw_rouge
```


## Usage
### Use the Script
```
usage: eval.py [-h] [-r REFERENCE] [-s SUBMISSION]

optional arguments:
  -h, --help            show this help message and exit
  -r REFERENCE, --reference REFERENCE
  -s SUBMISSION, --submission SUBMISSION
```

Example:
```
python eval.py -r public.jsonl -s submission.jsonl
{
  "rouge-1": {
    "f": 0.21999419163162043,
    "p": 0.2446195813913345,
    "r": 0.2137398792982201
  },
  "rouge-2": {
    "f": 0.0847583291303246,
    "p": 0.09419044877345074,
    "r": 0.08287844474014894
  },
  "rouge-l": {
    "f": 0.21017939117006337,
    "p": 0.25157090570020846,
    "r": 0.19404349000921203
  }
}
```


### Use Python Library
```
>>> from tw_rouge import get_rouge
>>> get_rouge('我是人', '我是一個人')
{'rouge-1': {'f': 0.7499999953125, 'p': 1.0, 'r': 0.6}, 'rouge-2': {'f': 0.33333332888888895, 'p': 0.5, 'r': 0.25}, 'rouge-l': {'f': 0.7499999953125, 'p': 1.0, 'r': 0.6}}
>>> get_rouge(['我是人'], [ '我是一個人'])
{'rouge-1': {'f': 0.7499999953125, 'p': 1.0, 'r': 0.6}, 'rouge-2': {'f': 0.33333332888888895, 'p': 0.5, 'r': 0.25}, 'rouge-l': {'f': 0.7499999953125, 'p': 1.0, 'r': 0.6}}
>>> get_rouge(['我是人'], ['我是一個人'], avg=False)
[{'rouge-1': {'f': 0.7499999953125, 'p': 1.0, 'r': 0.6}, 'rouge-2': {'f': 0.33333332888888895, 'p': 0.5, 'r': 0.25}, 'rouge-l': {'f': 0.7499999953125, 'p': 1.0, 'r': 0.6}}]
```


## Reference
[cccntu/tw_rouge](https://github.com/cccntu/tw_rouge)

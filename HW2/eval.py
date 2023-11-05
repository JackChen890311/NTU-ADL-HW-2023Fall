import json
import argparse
from tw_rouge import get_rouge


def eval(args):
    refs, preds = {}, {}

    with open(args.reference) as file:
        for line in file:
            line = json.loads(line)
            refs[line['id']] = line['title'].strip() + '\n'

    with open(args.submission) as file:
        for line in file:
            line = json.loads(line)
            preds[line['id']] = line['title'].strip() + '\n'

    keys =  refs.keys()
    refs = [refs[key] for key in keys]
    preds = [preds[key] for key in keys]

    score = get_rouge(preds, refs)
    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reference')
    parser.add_argument('-s', '--submission')
    args = parser.parse_args()
    score = eval(args)

    print(json.dumps(score, indent=2))
    print('Rouge 1 score:',score['rouge-1']['f'])
    print('Rouge 2 score:',score['rouge-2']['f'])
    print('Rouge L score:',score['rouge-l']['f'])
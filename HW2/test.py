import json
# from tw_rouge import get_rouge

if __name__ == '__main__':
    data = []
    with open('data/public.jsonl', 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    print(len(data))
    print(data[0])
    print(data[0].keys())
    # print(get_rouge('我是人', '我是一個人'))
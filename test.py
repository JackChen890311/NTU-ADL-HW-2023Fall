from datasets import load_dataset

def prepare_train_features(examples):
    # Dataset.arrow_dataset.Batch
    # Why can't I get the keys??? Hugging Face you kidding me?
    print(examples['id'][:10]) 
    
    # Manipulate with original features
    examples['question'] = [i + '???????' for i in examples['question']]
    print(examples['question'][:10])
    print(len(examples['question']))

    # we can create new features here
    examples['fuck'] = [i + '!!!!!' for i in examples['question']]
    print(examples['fuck'][:10])
    print(len(examples['fuck']))

    return examples # need to return this

if __name__ == '__main__':
    data_files = {
        'train': 'data/train.json',
        'valid': 'data/valid.json',
        # 'test': 'data/test.json',
    }

    # DataDict {'name': Dataset()}
    raw_dataset = load_dataset('json', data_files=data_files)
    print(raw_dataset)
    print(raw_dataset['train'].features)
    print(raw_dataset['train']['question'][:10])
    
    train_dataset = raw_dataset['train']
    print(train_dataset['answer'][:10])

    train_dataset = train_dataset.map(
            prepare_train_features,
            batched=True,
        )
    
    print(train_dataset.features)
    print(train_dataset['question'][:10])
    print(len(train_dataset['question']))
    
    print(train_dataset['fuck'][:10])
    print(len(train_dataset['fuck']))

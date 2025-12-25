import datasets

def preprocess_fn(example):
    context = example['context'].strip() + ' '
    question = '\n' + example['query'].strip() + ' '
    answer = example['target'].strip()
    return {'context': context, 'query': question, 'target': answer}

def preprocess_dataset(raw_dataset):
    return datasets.DatasetDict({
        'train': raw_dataset['train'].map(preprocess_fn, remove_columns=raw_dataset['train'].column_names),
        'valid': raw_dataset['test'].map(preprocess_fn, remove_columns=raw_dataset['test'].column_names)
        })
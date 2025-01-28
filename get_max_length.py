
from datasets import Dataset
import json
import numpy as np
from transformers import DebertaTokenizerFast

with open('./data/train_data.json') as f:
    data = json.load(f)

tokenizer = DebertaTokenizerFast.from_pretrained('microsoft/deberta-base', add_prefix_space=True)

ner_data = []
labels_list = []
for item in data:
    tokens = item['tokens']
    labels = item['Label']
    labels_list.append(labels)
    ner_data.append({
        'tokens': tokens,
        'ner_tags': labels
    })

dataset = Dataset.from_list(ner_data)

dataset_split = dataset.train_test_split(test_size=0.2)

train_dataset = dataset_split['train']
test_dataset = dataset_split['test']

def compute_token_lengths(examples):
    tokens = tokenizer(examples['tokens'], is_split_into_words=True, truncation=False)
    return {'length': [len(t) for t in tokens['input_ids']]}

tokenized_train_lengths = train_dataset.map(compute_token_lengths, batched=True)
tokenized_test_lengths = test_dataset.map(compute_token_lengths, batched=True)

train_lengths = tokenized_train_lengths['length']
print(f"Max sequence length in the train dataset: {max(train_lengths)}")
print(f"Average sequence length in the train dataset: {sum(train_lengths)/len(train_lengths)}")
print(f"95th percentile length in the train dataset: {np.percentile(train_lengths, 95)}")
print(f"99th percentile length in the train dataset: {np.percentile(train_lengths, 99)}")


"""
Max sequence length in the train dataset: 505
Average sequence length in the train dataset: 375.265
95th percentile length in the train dataset: 470.04999999999995 [This value is chosen in the training code line-51 in train.py]
99th percentile length in the train dataset: 496.01

"""
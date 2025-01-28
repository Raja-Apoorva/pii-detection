import json
from transformers import DebertaTokenizerFast
from transformers import DebertaForTokenClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

with open('./data/train_data.json') as f:
    data = json.load(f)

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

def flatten(xss):
    return [x for xs in xss for x in xs]

fl = flatten(labels_list)
label_list = np.unique(fl)

label2id = {label: i for i, label in enumerate(label_list)}
id2label = {id_: label for label, id_ in label2id.items()}

for item in ner_data:
    item['ner_tags'] = [label2id[label] for label in item['ner_tags']]

tokenizer = DebertaTokenizerFast.from_pretrained('microsoft/deberta-base', add_prefix_space=True)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples['tokens'],
        truncation=True,
        padding='max_length',  
        max_length=470,  
        is_split_into_words=True
    )
    
    word_ids = tokenized_inputs.word_ids()  

    label_ids = []
    for word_idx in word_ids:
        if word_idx is None: 
            label_ids.append(-100) 
        else:
            label_ids.append(examples['ner_tags'][word_idx])

    tokenized_inputs["labels"] = label_ids
    return tokenized_inputs


tokenized_dataset = [tokenize_and_align_labels(item) for item in ner_data]

model = DebertaForTokenClassification.from_pretrained(
    "microsoft/deberta-base",
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id,
    hidden_dropout_prob=0.1,  
    attention_probs_dropout_prob=0.1  
)



for name, param in model.deberta.named_parameters():
    if "layer.11" in name:  
        param.requires_grad = True
    else:
        param.requires_grad = False

dataset = Dataset.from_list(tokenized_dataset)

train_test_split = dataset.train_test_split(test_size=0.2) 

train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10
)

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [id2label[pred] for pred, label in zip(prediction, label) if label != -100]
        for prediction, label in zip(predictions, labels)
    ]

    accuracy = accuracy_score(true_labels, true_predictions)
    f1 = f1_score(true_labels, true_predictions)
    precision = precision_score(true_labels, true_predictions)
    recall = recall_score(true_labels, true_predictions)

    flat_true_labels = [label for sublist in true_labels for label in sublist]
    flat_true_predictions = [pred for sublist in true_predictions for pred in sublist]

    cm = confusion_matrix(flat_true_labels, flat_true_predictions, labels=list(id2label.values()))

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm.tolist()
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


# Count total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

model_param_summary = {
    "Total Parameters": total_params,
    "Trainable Parameters": trainable_params,
    "Non-trainable Parameters": total_params - trainable_params
}

print(model_param_summary)

trainer.train()


eval_results = trainer.evaluate()

print(eval_results)

model.save_pretrained("./results/ner_deberta_model_test")
tokenizer.save_pretrained("./results/ner_deberta_model_test")





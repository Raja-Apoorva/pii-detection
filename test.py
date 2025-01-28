import json
import numpy as np
from transformers import DebertaTokenizerFast
from transformers import DebertaForTokenClassification
from datasets import Dataset
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from transformers import Trainer, TrainingArguments
import plotly.figure_factory as ff

model = DebertaForTokenClassification.from_pretrained("./results/ner_deberta_model")
tokenizer = DebertaTokenizerFast.from_pretrained("./results/ner_deberta_model")

with open('./data/test_data.json') as f:
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

dataset = Dataset.from_list(tokenized_dataset)

training_args = TrainingArguments(
    output_dir="./test_results",
    per_device_eval_batch_size=8
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

    label_list = list(id2label.values()) 
    x = list(label_list)  
    y = list(label_list) 

    z_text = [[str(y) for y in x] for x in cm]

    fig = ff.create_annotated_heatmap(cm, x=x, y=y, annotation_text=z_text, colorscale='Blues')

    # Update layout for better readability
    fig.update_layout(
        title_text='Confusion Matrix',
        xaxis=dict(title='Predicted Label'),
        yaxis=dict(title='True Label'),
        autosize=False,
        width=700,
        height=700
    )

    fig.write_image("confusion_matrix_plotly.png")  

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
    eval_dataset=dataset,  
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

test_results = trainer.evaluate()

print("Test Results:")
print(test_results)

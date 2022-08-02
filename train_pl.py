import pandas as pd
import numpy as np
import datasets
from datasets.dataset_dict import Dataset
from pytorch_accelerated import Trainer
from sklearn.model_selection import train_test_split
from torch import nn, optim
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
import torch


bert_model = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(bert_model)


def preprocess_data(examples):
    # take a batch of texts
    text = examples["text"]
    # encode them
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)
    # add labels
    labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
    # create numpy array of shape (batch_size, num_labels)
    labels_matrix = np.zeros((len(text), len(labels)))
    # fill numpy array
    for idx, label in enumerate(labels):
        labels_matrix[:, idx] = labels_batch[label]

    encoding["labels"] = labels_matrix.tolist()

    return encoding


def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions,
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds,
        labels=p.label_ids)
    return result


data = pd.read_csv('DATASET/formated.csv')
# data = data.sample(1000, random_state=10)
data = data.iloc[:, :-1]
labels = list(data.columns[:-1])

train, test = train_test_split(data, test_size=0.2, random_state=0)
train_dataset = Dataset.from_dict(train)
test_dataset = Dataset.from_dict(test)
my_dataset_dict = datasets.DatasetDict({"train":train_dataset,"test":test_dataset})
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}
id2label = {idx:label for idx, label in enumerate(labels)}
encoded_dataset = my_dataset_dict.map(preprocess_data, batched=True,
                                      remove_columns=my_dataset_dict['train'].column_names)

encoded_dataset.set_format("torch")

model = AutoModelForSequenceClassification.from_pretrained(bert_model,
                                                           problem_type="multi_label_classification",
                                                           num_labels=len(labels),
                                                           id2label=id2label,
                                                           label2id=label2id)


#forward pass
outputs = model(input_ids=encoded_dataset['train']['input_ids'][0].unsqueeze(0),
                labels=encoded_dataset['train'][0]['labels'].unsqueeze(0))

loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
trainer = Trainer(
    model,
    loss_func=loss_func,
    optimizer=optimizer,
)
train_dataset = torch.tensor(train.values)
test_dataset = torch.tensor(test.values)
trainer.train(
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        num_epochs=3,
        per_device_batch_size=32,
    )
trainer.evaluate(
    dataset=test_dataset,
    per_device_batch_size=64,
)
# trainer.save_model("resources/model.pt")
#
# text = "অবশেষে জাতীয় পার্টি স্বীকার করলো তারা রাতের ভোটে বিরোধীদল হয়েছে! মুহাম্মদ রাশেদ খাঁন আগামী নির্বাচনে বিরোধীদল হতে মরিয়া"
# encoding = tokenizer(text, return_tensors="pt")
# encoding = {k: v.to(trainer.model.device) for k,v in encoding.items()}
# outputs = trainer.model(**encoding)
# logits = outputs.logits
# # apply sigmoid + threshold
# sigmoid = torch.nn.Sigmoid()
# probs = sigmoid(logits.squeeze().cpu())
# predictions = np.zeros(probs.shape)
# predictions[np.where(probs >= 0.5)] = 1
# # turn predicted id's into actual label names
# predicted_labels = [id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
# print(predicted_labels)

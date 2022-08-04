import numpy as np
from transformers import AutoTokenizer


class TextEncoder:

    def __init__(self, bert_model, labels):
        self.bert_model = bert_model
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)

    def preprocess_data(self, examples):
        # take a batch of texts
        text = examples["text"]
        # encode them
        encoding = self.tokenizer(text, padding="max_length", truncation=True, max_length=128)
        # add labels
        labels_batch = {k: examples[k] for k in examples.keys() if k in self.labels}
        # create numpy array of shape (batch_size, num_labels)
        labels_matrix = np.zeros((len(text), len(self.labels)))
        # fill numpy array
        for idx, label in enumerate(self.labels):
            labels_matrix[:, idx] = labels_batch[label]

        encoding["labels"] = labels_matrix.tolist()
        return encoding
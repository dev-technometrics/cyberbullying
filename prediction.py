import json
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer

model = AutoModelForSequenceClassification.from_pretrained("resources/model.pt")
bert_model = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(bert_model)
trainer = Trainer(model=model)
with open('resources/config.json') as file:
    config = json.load(file)
id2label = config['id2label']


class Predictor:
    id2label = id2label
    bert_model = bert_model
    def predict(self, texts, threshold=0.5):

        if isinstance(texts, str):
            texts = [texts]

        encoding = tokenizer(texts, return_tensors="pt")
        encoding = {k: v.to(trainer.model.device) for k,v in encoding.items()}
        outputs = trainer.model(**encoding)
        logits = outputs.logits
        # apply sigmoid + threshold
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits.squeeze().cpu())
        predictions = np.zeros(probs.shape)
        predictions[np.where(probs >= threshold)] = 1
        # turn predicted id's into actual label names
        predicted_labels = [id2label[str(idx)] for idx, label in enumerate(predictions) if label == 1.0]
        return predicted_labels, predictions

# texts = ["অবশেষে জাতীয় পার্টি স্বীকার করলো তারা রাতের ভোটে বিরোধীদল হয়েছে! মুহাম্মদ রাশেদ খাঁন আগামী নির্বাচনে বিরোধীদল হতে মরিয়া"]
# print(predict(texts))
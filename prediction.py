import json
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer

class Predictor:
    def __init__(self, bert_model):
        self.model_path = f"resources/model_{bert_model.replace('/', '_')}.pt"
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.bert_model = bert_model
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.trainer = Trainer(model=self.model)
        with open('resources/config.json') as file:
            config = json.load(file)
        self.id2label = config['id2label']

    def predict(self, texts, threshold=0.5):

        if isinstance(texts, str):
            texts = [texts]

        encoding = self.tokenizer(texts, return_tensors="pt")
        encoding = {k: v.to(self.trainer.model.device) for k,v in encoding.items()}
        outputs = self.trainer.model(**encoding)
        logits = outputs.logits
        # apply sigmoid + threshold
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits.squeeze().cpu())
        predictions = np.zeros(probs.shape)
        predictions[np.where(probs >= threshold)] = 1
        # turn predicted id's into actual label names
        predicted_labels = [self.id2label[str(idx)] for idx, label in enumerate(predictions) if label == 1.0]
        return predicted_labels, predictions

# texts = ["অবশেষে জাতীয় পার্টি স্বীকার করলো তারা রাতের ভোটে বিরোধীদল হয়েছে! মুহাম্মদ রাশেদ খাঁন আগামী নির্বাচনে বিরোধীদল হতে মরিয়া"]
# print(predict(texts))
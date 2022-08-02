import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer

model = AutoModelForSequenceClassification.from_pretrained("resources/model.pt")
bert_model = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(bert_model)
text = ["অবশেষে জাতীয় পার্টি স্বীকার করলো তারা রাতের ভোটে বিরোধীদল হয়েছে! মুহাম্মদ রাশেদ খাঁন আগামী নির্বাচনে বিরোধীদল হতে মরিয়া"]
trainer = Trainer(model=model)
# trainer.model = model.cuda()
# y = trainer.pr(text)
# print(y)
encoding = tokenizer(text, return_tensors="pt")
encoding = {k: v.to(trainer.model.device) for k,v in encoding.items()}
outputs = trainer.model(**encoding)
logits = outputs.logits
# apply sigmoid + threshold
sigmoid = torch.nn.Sigmoid()
probs = sigmoid(logits.squeeze().cpu())
predictions = np.zeros(probs.shape)
predictions[np.where(probs >= 0.5)] = 1
print(0)
# turn predicted id's into actual label names
# predicted_labels = [id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
# print(predicted_labels)
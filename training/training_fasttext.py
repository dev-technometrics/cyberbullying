import json

import fasttext
import numpy as np

class FasttextTrainer:
    def __init__(self, fasttext_params):
        self.fasttext_params = fasttext_params
    def train_supervised(self, model_path):
        model = fasttext.train_supervised(**self.fasttext_params)
        print('vocab size: ', len(model.words))
        print('label size: ', len(model.labels))
        print('example vocab: ', model.words[:5])
        print('example label: ', model.labels[:5])
        text = 'অবশেষে জাতীয় পার্টি স্বীকার করলো তারা রাতের ভোটে বিরোধীদল হয়েছে! মুহাম্মদ রাশেদ খাঁন আগামী নির্বাচনে বিরোধীদল হতে মরিয়া'
        print(model.predict(text, k=3))
        model.save_model(f"{model_path}")
class FasttextPredictor():
    def __init__(self,model_filepath):
        self.model_filepath = model_filepath
        self.classifier = fasttext.load_model(self.model_filepath)
        with open('resources/config.json') as file:
            config = json.load(file)
        self.label2id = config['label2id']
        self.id2label = config['id2label']
        self.label_dict = {}
        for l in self.label2id:
            self.label_dict[f'{l.replace(", ", "_").replace(": ", "_").replace(" ", "_").replace(" - ", "_").replace("/", "_")}'] = l

    def predict(self, text):
        labels = self.classifier.predict(text, k=len(self.label2id))
        probs = labels[1]
        predictions = np.zeros(probs.shape)
        predictions[np.where(probs >= 0.333)] = 1
        return predictions

    def get_prediction_array(self, text):
        predictions = self.predict(text)
        labels = [self.id2label[str(idx)] for idx, label in enumerate(predictions) if label == 1.0]
        return predictions, labels
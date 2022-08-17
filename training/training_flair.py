import json
import numpy as np
import torch
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from flair.data import Corpus
from flair.datasets import ClassificationCorpus
from flair.data import Sentence
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer

from settings import DIR_RESOURCES, DIR_MODEL_PYTORCH
from utils import make_dir_if_not_exists


class FlairTrainer:

    def __init__(self, label_type):
        self.label_type = label_type

    def train(self,
              pretrained_model,
              is_multi_label,
              model_path,
              data_folder):

        self.corpus: Corpus = ClassificationCorpus(data_folder,
                                                   test_file='test.txt',
                                                   dev_file='test.txt',
                                                   train_file='train.txt',
                                                   label_type=self.label_type,
                                                   )
        self.label_dict = self.corpus.make_label_dictionary(label_type=self.label_type)
        self.document_embeddings = TransformerDocumentEmbeddings(pretrained_model, fine_tune=True)
        self.classifier = TextClassifier(self.document_embeddings,
                                         label_dictionary=self.label_dict,
                                         label_type=self.label_type, multi_label=is_multi_label)
        self.trainer = ModelTrainer(self.classifier, self.corpus)
        self.model_path = f'{model_path}/'
        make_dir_if_not_exists(self.model_path)
        self.trainer.fine_tune(self.model_path,
                          learning_rate=5.0e-5,
                          mini_batch_size=4,
                          max_epochs=10,
                          )

    def predict(self, model_path, text):
        classifier = TextClassifier.load(model_path)
        sentence = Sentence(text)
        classifier.predict(sentence)
        return sentence.labels

class FlairPredictor(FlairTrainer):
    def __init__(self, model_path):
        super().__init__('topic')
        self.model_path = model_path
        self.classifier = TextClassifier.load(self.model_path)
        with open( f"{DIR_RESOURCES}config.json") as file:
            config = json.load(file)
        self.label2id = config['label2id']
        self.label_dict = {}
        for l in self.label2id:
            self.label_dict[f'{l.replace(", ", "_").replace(": ", "_").replace(" ", "_").replace(" - ", "_").replace("/", "_")}'] = l
    def predict(self, text):
        sentence = Sentence(text)
        self.classifier.predict(sentence)
        return sentence.labels

    def get_predicted_labels(self, text):
        labels = self.predict(text)
        preds = []
        for label in labels:
            preds.append(label.value)
        return preds
    def get_prediction_array(self, text):
        labels = self.get_predicted_labels(text)
        # if labels:
        #     print(0)
        preds = [0 for i in range(len(self.label2id))]
        for label in labels:
            y = f'{label.replace(", ", "_").replace(": ", "_").replace(" ", "_").replace(" - ", "_").replace("/", "_")}'
            l = self.label_dict[y]
            preds[self.label2id[l]] = 1
        return preds

class PytorchModelPredictor():
    def __init__(self, bert_model, model_path):
        self.model_path = model_path
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.bert_model = bert_model
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.trainer = Trainer(model=self.model)
        with open( f"{DIR_RESOURCES}config.json") as file:
            config = json.load(file)
        self.id2label = config['id2label']

    def predict(self, texts, threshold=0.5):
        if isinstance(texts, str):
            texts = [texts]
        encoding = self.tokenizer(texts, return_tensors="pt")
        encoding = {k: v.to(self.trainer.model.device) for k, v in encoding.items()}
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
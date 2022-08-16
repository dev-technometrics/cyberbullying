import json

from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from flair.data import Corpus
from flair.datasets import ClassificationCorpus
from flair.data import Sentence

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
                          max_epochs=2,
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
        with open('resources/config.json') as file:
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
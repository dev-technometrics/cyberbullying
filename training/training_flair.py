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
                                                   test_file='test_fasttext_data.txt',
                                                   dev_file='test_fasttext_data.txt',
                                                   train_file='train_fasttext_data.txt',
                                                   label_type=self.label_type,
                                                   )
        self.label_dict = self.corpus.make_label_dictionary(label_type=self.label_type)
        self.document_embeddings = TransformerDocumentEmbeddings(pretrained_model, fine_tune=True)
        self.classifier = TextClassifier(self.document_embeddings,
                                         label_dictionary=self.label_dict,
                                         label_type=self.label_type, multi_label=is_multi_label)
        self.trainer = ModelTrainer(self.classifier, self.corpus)
        self.model_path = f'{model_path}/{pretrained_model.replace("/", "_")}'
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
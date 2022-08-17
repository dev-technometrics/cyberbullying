import pandas as pd
import numpy as np
import datasets
from datasets.dataset_dict import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch

from accuracy_calculation.accuracy_flair import calculate_accuracy_flair, calculate_accuracy_fasttext, \
    calculate_accuracy_pytorch
from performence_calculation.metrics_calculation import PerformenceCalculator
from preprocessing.cleaning import TextCleaner
from preprocessing.data_preparation import DataPreparation
from preprocessing.encoding import TextEncoder
from settings import MODEL_BERT_MULTILANGUAL_CASED, MODEL_BERT_CESBUETNLP, MODEL_BERT_MONSOON_NLP, \
    MODEL_BERT_SAGORSARKAR, MODEL_BERT_INDIC_NER, MODEL_BERT_NURALSPACE, MODEL_BERT_INDIC_HATE_SPEECH, \
    MODEL_BERT_NEUROPARK_SAHAJ_NER, MODEL_BERT_NEUROPARK_SAHAJ, DIR_DATASET, DIR_RESOURCES, DIR_MODEL_FLAIR, \
    DIR_DATASET_FASTTEXT, DIR_MODEL_FASTTEXT, DIR_MODEL_PYTORCH
from training.training_fasttext import FasttextTrainer
from training.training_flair import FlairTrainer

def train_pytorch(bert_models, train, test):
    text_cleaner = TextCleaner()
    data = train
    data['text'] = data['text'].apply(text_cleaner.clean_text_bn)
    data = data.iloc[:, :-1]
    train = train.iloc[:, :-1]
    test = test.iloc[:, :-1]
    labels = list(data.columns[:-1])
    train_dataset = Dataset.from_dict(train)
    test_dataset = Dataset.from_dict(test)
    my_dataset_dict = datasets.DatasetDict({"train": train_dataset, "test": test_dataset})
    id2label = {idx: label for idx, label in enumerate(labels)}
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for idx, label in enumerate(labels)}
    batch_size = 32
    metric_name = "f1"
    epoch = 10
    args = TrainingArguments(
        f"bert-finetuned-multi-label-topic",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epoch,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        # push_to_hub=True,
        # no_cuda=True
    )
    performence_calculator = PerformenceCalculator()

    for bert_model in bert_models:
        print('************************************')
        print(f'Started {bert_model} model training')
        print('************************************')
        text_encoder = TextEncoder(bert_model, labels)
        encoded_dataset = my_dataset_dict.map(text_encoder.preprocess_data, batched=True,
                                              remove_columns=my_dataset_dict['train'].column_names)
        encoded_dataset.set_format("torch")
        model = AutoModelForSequenceClassification.from_pretrained(bert_model,
                                                                   problem_type="multi_label_classification",
                                                                   num_labels=len(labels),
                                                                   id2label=id2label,
                                                                   label2id=label2id,
                                                                   ignore_mismatched_sizes=True)

        #forward pass
        outputs = model(input_ids=encoded_dataset['train']['input_ids'][0].unsqueeze(0),
                        labels=encoded_dataset['train'][0]['labels'].unsqueeze(0))

        trainer = Trainer(
            model,
            args,
            train_dataset=encoded_dataset["train"],
            eval_dataset=encoded_dataset["test"],
            tokenizer=text_encoder.tokenizer,
            compute_metrics=performence_calculator.compute_metrics
        )
        trainer.train()
        trainer.evaluate()
        trainer.save_model(f"{DIR_MODEL_PYTORCH}{bert_model.replace('/', '_')}.pt")

        text = "অবশেষে জাতীয় পার্টি স্বীকার করলো তারা রাতের ভোটে বিরোধীদল হয়েছে! মুহাম্মদ রাশেদ খাঁন আগামী নির্বাচনে বিরোধীদল হতে মরিয়া"
        encoding = text_encoder.tokenizer(text, return_tensors="pt")
        encoding = {k: v.to(trainer.model.device) for k,v in encoding.items()}
        outputs = trainer.model(**encoding)
        logits = outputs.logits
        # apply sigmoid + threshold
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits.squeeze().cpu())
        predictions = np.zeros(probs.shape)
        predictions[np.where(probs >= 0.5)] = 1
        # turn predicted id's into actual label names
        predicted_labels = [id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
        print(predicted_labels)

    calculate_accuracy_pytorch(bert_models, test)

def train_flair(bert_models, test):
    for bert_model in bert_models:
        print('************************************')
        print(f'Started {bert_model} model training')
        print('************************************')

        try:
            flair_trainer = FlairTrainer(label_type='topic')

            model_path = f'{DIR_MODEL_FLAIR}{bert_model.replace("/", "_")}'

            flair_trainer.train(pretrained_model=bert_model,
                                is_multi_label=True,
                                model_path=model_path,
                                data_folder=DIR_DATASET_FASTTEXT)

            labels = flair_trainer.predict(model_path=f'{model_path}/final-model.pt',
                text='অবশেষে জাতীয় পার্টি স্বীকার করলো তারা রাতের ভোটে বিরোধীদল হয়েছে! মুহাম্মদ রাশেদ খাঁন আগামী নির্বাচনে বিরোধীদল হতে মরিয়া')
            print(labels)
        except Exception as e:
            print('************************************')
            print(f'error {bert_model} model training')
            print(e)
            print('************************************')

    calculate_accuracy_flair(bert_models, test)

def train_fastext(test):
    fasttext_params = {
        'input': f'{DIR_DATASET_FASTTEXT}train.txt',
        'lr': 0.1,
        'lrUpdateRate': 1000,
        'thread': 8,
        'epoch': 100,
        'wordNgrams': 2,
        'dim': 1000,
        'verbose': 5,
        'loss': 'ova'
    }
    model_filepath = f'{DIR_MODEL_FASTTEXT}final-model_pretrained.bin'
    fasttext_trainer = FasttextTrainer(fasttext_params)
    fasttext_trainer.train_supervised(model_filepath)
    calculate_accuracy_fasttext('fasttext-pretrained', model_filepath, test)

if __name__ == "__main__":

    data_preparation = DataPreparation()
    data = pd.read_csv(f'{DIR_DATASET}formated.csv')
    # data = data.sample(100, random_state=10)
    train, test = train_test_split(data, test_size=0.2, random_state=0)
    # data_preparation.prepare_for_fasttext(train, f'{DIR_DATASET_FASTTEXT}train.txt')
    # data_preparation.prepare_for_fasttext(test, f'{DIR_DATASET_FASTTEXT}test.txt')


    bert_models = [MODEL_BERT_MULTILANGUAL_CASED, MODEL_BERT_CESBUETNLP, MODEL_BERT_MONSOON_NLP,
                   MODEL_BERT_SAGORSARKAR, MODEL_BERT_INDIC_NER, MODEL_BERT_NURALSPACE, MODEL_BERT_INDIC_HATE_SPEECH,
                   MODEL_BERT_NEUROPARK_SAHAJ_NER, MODEL_BERT_NEUROPARK_SAHAJ]
    # bert_models = [MODEL_BERT_NEUROPARK_SAHAJ]

    train_flair(bert_models, test)
    # train_fastext(test)
    # train_pytorch(bert_models, train, test)

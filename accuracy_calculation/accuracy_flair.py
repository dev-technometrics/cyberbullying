import numpy as np
import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from preprocessing.cleaning import TextCleaner
from settings import DIR_RESOURCES, DIR_PERFORMENCE_REPORT_FLAIR, DIR_PERFORMENCE_REPORT_FASTTEXT, DIR_MODEL_FASTTEXT, \
    DIR_MODEL_PYTORCH, DIR_PERFORMENCE_REPORT_PYTORCH, DIR_MODEL_FLAIR
from training.training_fasttext import FasttextPredictor
from training.training_flair import FlairTrainer, FlairPredictor, PytorchModelPredictor


def calculate_accuracy_flair(bert_models, data):
    text_cleaner = TextCleaner()
    data = data.iloc[:, :-1]
    data['text'] = data['text'].apply(text_cleaner.clean_text_bn)
    texts = data.iloc[:, -1]
    for bert_model in bert_models:
        print('************************************')
        print(f'Started {bert_model} model accuracy calculation')
        print('************************************')
        try:
            model_path = f'{DIR_MODEL_FLAIR}{bert_model.replace("/", "_")}'
            flair_predictor = FlairPredictor(f'{model_path}/final-model.pt')
            predictions = []
            for text in texts:
                prediction = flair_predictor.get_prediction_array(text=text)
                predictions.append(prediction)
            y_pred = np.array(predictions)
            y_true = data.iloc[:, :-1].values
            y_true = y_true * 1
            cm = multilabel_confusion_matrix(y_true, y_pred)
            print(cm)
            with open(f"{DIR_PERFORMENCE_REPORT_FLAIR}{bert_model.replace('/', '_')}_cm.txt", 'w') as cm_file:
                cm_file.write(str(cm))
            cr = classification_report(y_true, y_pred, target_names=flair_predictor.label2id.keys())
            print(cr)
            with open(f"{DIR_PERFORMENCE_REPORT_FLAIR}{bert_model.replace('/', '_')}_cr.txt", 'w') as cr_file:
                cr_file.write(str(cr))

        except Exception as e:
            print('************************************')
            print(f'error {bert_model} model accuracy calculation')
            print(e)
            print('************************************')


def calculate_accuracy_fasttext(model_name, model_filepath, data):
    text_cleaner = TextCleaner()
    data = data.iloc[:, :-1]
    data['text'] = data['text'].apply(text_cleaner.clean_text_bn)
    texts = data.iloc[:, -1]
    print('************************************')
    print(f'Started {model_name} model accuracy calculation')
    print('************************************')
    try:
        flair_predictor = FasttextPredictor(model_filepath)
        predictions = []
        for text in texts:
            prediction, label = flair_predictor.get_prediction_array(text=text)
            predictions.append(prediction)
        y_pred = np.array(predictions)
        y_true = data.iloc[:, :-1].values
        y_true = y_true * 1
        cm = multilabel_confusion_matrix(y_true, y_pred)
        print(cm)
        with open(f"{DIR_PERFORMENCE_REPORT_FASTTEXT}{model_name.replace('/', '_')}_cm.txt", 'w') as cm_file:
            cm_file.write(str(cm))
        cr = classification_report(y_true, y_pred, target_names=flair_predictor.label2id.keys())
        print(cr)
        with open(f"{DIR_PERFORMENCE_REPORT_FASTTEXT}{model_name.replace('/', '_')}_cr.txt", 'w') as cr_file:
            cr_file.write(str(cr))

    except Exception as e:
        print(e)


def calculate_accuracy_pytorch(bert_models, data):
    text_cleaner = TextCleaner()
    data = data.iloc[:, :-1]
    data['text'] = data['text'].apply(text_cleaner.clean_text_bn)
    texts = data.iloc[:, -1]
    for bert_model in bert_models:
        print('************************************')
        print(f'Started {bert_model} model accuracy calculation')
        print('************************************')
        try:
            model_path = f"{DIR_MODEL_PYTORCH}{bert_model.replace('/', '_')}.pt"
            pytorch_predictor = PytorchModelPredictor(bert_model, f'{model_path}')
            predictions = []
            for text in texts:
                predicted_labels, prediction = pytorch_predictor.predict(text)
                predictions.append(prediction)
            y_pred = np.array(predictions)
            y_true = data.iloc[:, :-1].values
            y_true = y_true * 1
            cm = multilabel_confusion_matrix(y_true, y_pred)
            print(cm)
            with open(f"{DIR_PERFORMENCE_REPORT_PYTORCH}{bert_model.replace('/', '_')}_cm.txt", 'w') as cm_file:
                cm_file.write(str(cm))
            cr = classification_report(y_true, y_pred, target_names=pytorch_predictor.id2label.values())
            print(cr)
            with open(f"{DIR_PERFORMENCE_REPORT_PYTORCH}{bert_model.replace('/', '_')}_cr.txt", 'w') as cr_file:
                cr_file.write(str(cr))

        except Exception as e:
            print(e)

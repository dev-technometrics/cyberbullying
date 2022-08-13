import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, classification_report

from prediction import Predictor
from preprocessing.cleaning import TextCleaner
from settings import MODEL_BERT_MULTILANGUAL_CASED, DIR_PERFORMENCE_REPORT, DIR_PERFORMENCE_REPORT_PYTORCH, \
    MODEL_BERT_CESBUETNLP, MODEL_BERT_MONSOON_NLP, MODEL_BERT_SAGORSARKAR, MODEL_BERT_INDIC_NER, MODEL_BERT_NURALSPACE, \
    MODEL_BERT_INDIC_HATE_SPEECH, MODEL_BERT_NEUROPARK_SAHAJ_NER, MODEL_BERT_NEUROPARK_SAHAJ


def main(bert_models):

    data = pd.read_csv('DATASET/formated.csv')
    # data = data.sample(10, random_state=0)
    text_cleaner = TextCleaner()
    data = data.iloc[:, :-1]
    train, test = train_test_split(data, test_size=0.2, random_state=0)
    test['text'] = test['text'].apply(text_cleaner.clean_text_bn)
    texts = test.iloc[:, -1]

    for bert_model in bert_models:
        print('************************************')
        print(f'Started {bert_model} model accuracy calculation')
        print('************************************')
        predictions = []
        predictor = Predictor(bert_model)
        for text in texts:
            predicted_labels, prediction = predictor.predict(text)
            predictions.append(prediction)
        y_pred = np.array(predictions)
        y_true = test.iloc[:, :-1].values
        y_true = y_true*1
        cm = multilabel_confusion_matrix(y_true, y_pred)
        print(cm)
        with open(f"{DIR_PERFORMENCE_REPORT_PYTORCH}{bert_model.replace('/', '_')}_cm.txt", 'w') as cm_file:
            cm_file.write(str(cm))
        cr = classification_report(y_true, y_pred,target_names=predictor.id2label.values())
        print(cr)
        with open(f"{DIR_PERFORMENCE_REPORT_PYTORCH}{bert_model.replace('/', '_')}_cr.txt", 'w') as cr_file:
            cr_file.write(str(cr))

if __name__ == "__main__":

    bert_models = [MODEL_BERT_MULTILANGUAL_CASED, MODEL_BERT_CESBUETNLP, MODEL_BERT_MONSOON_NLP,
                   MODEL_BERT_SAGORSARKAR, MODEL_BERT_INDIC_NER, MODEL_BERT_NURALSPACE, MODEL_BERT_INDIC_HATE_SPEECH,
                   MODEL_BERT_NEUROPARK_SAHAJ_NER, MODEL_BERT_NEUROPARK_SAHAJ]

    # bert_models = [MODEL_BERT_MULTILANGUAL_CASED]

    main(bert_models)



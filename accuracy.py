import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, classification_report

from prediction import Predictor
from preprocessing.cleaning import TextCleaner

data = pd.read_csv('DATASET/formated.csv')
# data = data.sample(100, random_state=10)
data = data.iloc[:, :-1]
train, test = train_test_split(data, test_size=0.2, random_state=0)
texts = test.iloc[:, -1]
predictions = []
predictor = Predictor()
text_cleaner = TextCleaner()
for text in texts:
    predicted_labels, prediction = predictor.predict(text_cleaner.clean_text_bn(text))
    predictions.append(prediction)
y_pred = np.array(predictions)
y_true = test.iloc[:, :-1].values
y_true = y_true*1
cm = multilabel_confusion_matrix(y_true, y_pred)
print(cm)
print(classification_report(y_true, y_pred,target_names=predictor.id2label.values()))



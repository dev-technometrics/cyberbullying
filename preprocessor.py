import re
import pandas as pd
from sklearn.model_selection import train_test_split

from settings import DIR_RESOURCES


class PreProcessor():
    def cleaning_documents(self, articles):
        # remove non bangla text
        news = "".join(i for i in articles if i in [".","।"] or 2432 <= ord(i) <= 2559 or ord(i)== 32)
        # remove space
        news = news.replace('\n',' ')
        # remove unnecessary punctuation
        news = re.sub('[^\u0980-\u09FF]',' ',str(news))
        # remove stopwords
        stp = open(DIR_RESOURCES + '/bangla_stopwords.txt','r',encoding='utf-8').read().split()
        result = news.split()
        news = [word.strip() for word in result if word not in stp ]
        news =" ".join(news)
        return news

    def read_data(self):
        data = pd.read_csv('DATASET/formated.csv')
        data = data.sample(300, random_state=10)
        data = data.iloc[:, :-1]
        self.labels = list(data.columns[:-1])
        train, test = train_test_split(data, test_size=0.2, random_state=0)
        return train, test

    def get_label_info(self):

        id2label = {idx: label for idx, label in enumerate(self.labels)}
        label2id = {label: idx for idx, label in enumerate(self.labels)}
        id2label = {idx: label for idx, label in enumerate(self.labels)}

        return self.labels, id2label, label2id
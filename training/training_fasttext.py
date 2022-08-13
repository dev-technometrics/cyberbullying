import pandas as pd
from preprocessing.cleaning import TextCleaner

data = pd.read_csv('../DATASET/formated.csv')
text_cleaner = TextCleaner()
data['text'] = data['text'].apply(text_cleaner.clean_text_bn)
# data = data.iloc[:1000, :]
columns = data.columns[:-2]
texts = ''
for i in range(len(data)):
    print('*************************************************')
    txt = ''
    for column in columns:
        if data.loc[i][column]:
            txt += f'__label__{column.replace(", ", "_").replace(": ", "_").replace(" ", "_").replace(" - ", "_")} '
    txt += data.loc[i]['text']
    texts+= f'{txt}\n'
with open('../DATASET/fasttext_data.txt', 'w') as file:
    file.write(texts)

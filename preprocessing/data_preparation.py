from preprocessing.cleaning import TextCleaner

class DataPreparation:

    def __init__(self):
        self.text_cleaner = TextCleaner()

    def prepare_for_fasttext(self, data, output_filepath, text_column, label_column):
        data[text_column] = data[text_column].apply(self.text_cleaner.clean_text_bn)
        data = data.reset_index()
        columns = data.columns[1:-2]
        texts = ''
        for i in range(len(data)):
            try:
                print('*************************************************')
                txt = ''
                # for column in columns:
                #         if data.loc[i][column]:
                #             txt += f'__label__{column.replace(", ", "_").replace(": ", "_").replace(" ", "_").replace(" - ", "_")} '

                txt +=  f'__label__{data.loc[i][label_column].replace(", ", "_").replace(": ", "_").replace(" ", "_").replace(" - ", "_")} '
                txt += data.loc[i][text_column]
                texts+= f'{txt}\n'
            except KeyError as e:
                print(e)
        with open(f'{output_filepath}', 'w') as file:
            file.write(texts)

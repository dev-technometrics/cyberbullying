import os
def make_dir_if_not_exists(file_path):
    dirs = file_path.split('/')
    if dirs:
        path = ''
        for dir in dirs:
            if dir:
                path = path + dir + '/'
                if not os.path.exists(path):
                    os.mkdir(path)

MODEL_BIDIRECTIONAL_GRU = "bi_gru"
MODEL_CNN_BIDIRECTIONAL_LSTM = "cnn_bi_lstm"
MODEL_FASTTEXT_DEEP_ANN = "bi_lstm"
MODEL_FASTTEXT_SIMPLE = "simple"
MODEL_ML = "LinearSVC"

MODEL_BERT_MULTILANGUAL_CASED = 'bert-base-multilingual-cased'
MODEL_BERT_CESBUETNLP = 'csebuetnlp/banglabert'
MODEL_BERT_MONSOON_NLP = 'monsoon-nlp/bangla-electra'
MODEL_BERT_SAGORSARKAR = 'sagorsarker/bangla-bert-base'
MODEL_BERT_INDIC_NER = 'ai4bharat/IndicNER'
MODEL_BERT_NURALSPACE = 'neuralspace-reverie/indic-transformers-bn-distilbert'
MODEL_BERT_INDIC_HATE_SPEECH = 'Hate-speech-CNERG/indic-abusive-allInOne-MuRIL'
MODEL_BERT_NEUROPARK_SAHAJ_NER = 'neuropark/sahajBERT-NER'
MODEL_BERT_NEUROPARK_SAHAJ = 'neuropark/sahajBERT'

MODEL_SELECTED = MODEL_FASTTEXT_DEEP_ANN

DIR_BASE = f'{os.path.dirname(os.path.realpath(__file__))}/'
DIR_DATASET = 'DATASET/'
DIR_REPORT = 'REPORT/'
DIR_EDA = 'EDA/'
DIR_IMAGES_EDA = DIR_REPORT + 'IMAGES/EDA/'
DIR_IMAGES_HISTORY = DIR_REPORT + 'IMAGES/HISTORY/'
DIR_PERFORMENCE_REPORT = DIR_REPORT + 'PERFORMENCE/'
DIR_PERFORMENCE_REPORT_PYTORCH = DIR_PERFORMENCE_REPORT + 'PYTORCH/'
DIR_RESOURCES = DIR_BASE + 'resources/'
for i in [DIR_DATASET, DIR_REPORT, DIR_EDA, DIR_IMAGES_EDA, DIR_IMAGES_HISTORY, DIR_PERFORMENCE_REPORT,
          DIR_PERFORMENCE_REPORT_PYTORCH, DIR_RESOURCES]:
    make_dir_if_not_exists(i)

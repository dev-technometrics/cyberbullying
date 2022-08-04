import re

from settings import DIR_RESOURCES, DIR_BASE


class TextCleaner():

    def clean_text_bn(self, text):

        text = text.replace('\n', ' ')
        # remove unnecessary punctuation
        text = re.sub('[^\u0980-\u09FF]', ' ', str(text))
        text = re.sub(r'[~^০-৯]', '', text)
        whitespace = re.compile(u"[\s\u0020\u00a0\u1680\u180e\u202f\u205f\u3000\u2000-\u200a]+", re.UNICODE)
        bangla_fullstop = u"\u0964"
        punctSeq = u"['\"“”‘’]+|[.?!,…]+|[:;]+"
        punc = u"[(),$%^&*+={}\[\]:\"|\'\~`<>/,¦!?½£¶¼©⅐⅑⅒⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞⅟↉¤¿º;-]+"
        text = whitespace.sub(" ", text).strip()
        text = re.sub(punctSeq, " ", text)
        text = re.sub(bangla_fullstop, " ", text)
        text = re.sub(punc, " ", text)

        # remove stopwords
        result = text.split()
        stp = open(f'{DIR_RESOURCES}bangla_stopwords.txt', 'r', encoding='utf-8').read().split()
        try:
            stp_custom = open(f'{DIR_RESOURCES}bangla_stopwords_custom.txt', 'r',
                              encoding='utf-8').read().split()
            stp = stp + stp_custom
        except Exception as e:
            print(e)
        text = [word.strip() for word in result if word not in stp]
        text = " ".join(text)
        return text
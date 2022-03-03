import string
from typing import Text
from unicodedata import name
import re
import numpy as np 
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')

from nltk.corpus import stopwords
import sklearn 


class TextProcessor:

    def __init__(self,text):
        self.text = text
        self.stopwords = stopwords.words('english')
        #Keeping the special characters, we re-format the punctuation
        self.special_chars = re.compile('[{}]'.format(re.escape(string.punctuation)))

    def _iterator(self):
        for text in self.text:
            yield text

    def Preprocess_Text(self):
        for text in self._iterator():
            #Token creation 
            tokens = nltk.word_tokenize(text)
            tokens = [token.strip().lower() for token in tokens]

            #Stopword & Punctuation Removal 
            cleaned_tokens = [token for token in tokens if token not in self.stopwords]
            token_filters = filter(None,[self.special_chars.sub(' ', token) for token in cleaned_tokens])
            new_text = ' '.join(token_filters)
            return new_text

if __name__ == '__main__':
    tp = TextProcessor('The name of the first President is George Washington.')
    tp.Preprocess_Text()

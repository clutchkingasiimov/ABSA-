import string
from typing import Text
import re
import nltk
import spacy 
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
        self.nlp = spacy.load("en_core_web_sm")
        self.processed_text = []

    def _iterator(self):
        for text in self.text:
            yield text

    def Preprocess_Text(self):
        for text in self._iterator():
            #Remove numbers from the string 
            cleaned_text = re.sub(r'\d+', '', text)

            #Token creation & Lemmatization
            doc = self.nlp(cleaned_text)
            tokens = [token.lemma_ for token in doc]
            tokens = [token.strip().lower() for token in tokens]

            #Stopword & Punctuation Removal 
            cleaned_tokens = [token for token in tokens if token not in self.stopwords]
            token_filters = filter(None,[self.special_chars.sub(' ', token) for token in cleaned_tokens])
            new_text = ' '.join(token_filters)

            #Whitespace removal between pre-existing punctuations & stopwords 
            new_text = " ".join(new_text.split())
            self.processed_text.append(new_text)
        return self.processed_text


if __name__ == '__main__':
    texts = ['The name of the first President is George Washington',
    'He became the president in the year 1776',
    'Russia will continue invading Ukraine for quite some time, supposedly 2 years more',
    'NATO continues to sanction Russia in support of defending Ukraine, the sanctions being going on for 2 weeks now.']
    tp = TextProcessor(texts)
    tp.Preprocess_Text()

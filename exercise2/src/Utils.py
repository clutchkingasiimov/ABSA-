import pandas as pd
import string
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.tokenize import sent_tokenize, word_tokenize 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
from nltk.tag import pos_tag

import itertools

import mxnet as mx
from bert_embedding import BertEmbedding


def load_data(path):
  df = pd.read_csv(path, delimiter='\t', header=None)
  df = df.rename(columns={
    0:'polarity',
    1:'aspect_cat',
    2:'target_term',
    3:'char_offset',
    4:'sentence'
})
  df['polarity_2'] = df['polarity'].map({'positive':0,'negative':1,'neutral':2})
  
  return df


def NormalizeWithPOS(text):
    # Lemmatization & Stemming according to POS tagging

    word_list = word_tokenize(text)
    rev = []
    lemmatizer = WordNetLemmatizer() 
    stemmer = PorterStemmer() 
    for word, tag in pos_tag(word_list):
        if tag.startswith('J'):
            w = lemmatizer.lemmatize(word, pos='a')
        elif tag.startswith('V'):
            w = lemmatizer.lemmatize(word, pos='v')
        elif tag.startswith('N'):
            w = lemmatizer.lemmatize(word, pos='n')
        elif tag.startswith('R'):
            w = lemmatizer.lemmatize(word, pos='r')
        else:
            w = word
        w = stemmer.stem(w)
        rev.append(w)
    review = ' '.join(rev)
    return review


def cleanText(text):
    
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'re", " are", text)

    # if embedding is not 'BERT':
    #     text = re.sub(r"[0-9]+", ' ', text)
    #     text = re.sub(r"-", ' ', text)
    
    text = text.strip().lower()
    
    
    # if embedding is not 'BERT':
    #     # Remove other contractions
    #     text = re.sub(r"'", ' ', text)
    
    # Replace punctuations with space
    # if embedding is 'BERT': # save ! ? . for end of the sentence detection [,/():;']
    filters='"#$%&*+<=>@[\\]^_`{|}~\t\n'
    text = re.sub(r'\!+', '!', text)
    text = re.sub(r'\?+', '?', text)
    # else:
    #     filters='!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    # translate_dict = dict((i, " ") for i in filters)
    # translate_map = str.maketrans(translate_dict)
    # text = text.translate(translate_map)
    
    # if embedding is 'BERT':
    text = re.sub(r'\( *\)', ' ', text)

    # if embedding is not 'BERT':
    text = ' '.join([w for w in text.split() if len(w)>1])

    # Replace multiple space with one space
    text = re.sub(' +', ' ', text)
    
    text = ''.join(text)

    return text



def embeddToBERT(text):
    
    ctx = mx.cpu()
    bert = BertEmbedding(ctx=ctx)
    
    sentences = re.split('!|\?|\.',text)
    sentences = list(filter(None, sentences)) 

    # if bert_version == 'WORD':
    result = bert(sentences, 'avg') # avg is refer to handle OOV

    bert_vocabs_of_sentence = []
    for sentence in range(len(result)):
        for word in range(len(result[sentence][1])):
            bert_vocabs_of_sentence.append(result[sentence][1][word])
    feature = [mean(x) for x in zip(*bert_vocabs_of_sentence)]

    # elif bert_version == 'SENTENCE':
    #     result = bert_transformers.encode(sentences)
    #     feature = [mean(x) for x in zip(*result)]
  
    return feature


def mean(z): # used for BERT (word version) and Word2Vec
    return sum(itertools.chain(z))/len(z)
    
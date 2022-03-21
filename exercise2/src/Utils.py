import pandas as pd
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Text
import re
import nltk
import spacy

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
import sklearn

from torchtext import data



def load_data(path):
  df = pd.read_csv(path, delimiter='\t', header=None)
  df = df.rename(columns={
    0:'polarity',
    1:'aspect_cat',
    2:'target_term',
    3:'char_offset',
    4:'sentence'
})
  return df

class TextProcessor:

    def __init__(self, text):
      self.text = text
      self.stopwords = stopwords.words('english')
      # Keeping the special characters, we re-format the punctuation
      self.special_chars = re.compile('[{}]'.format(re.escape(string.punctuation)))
      self.nlp = spacy.load("en_core_web_sm")
      self.processed_text = []

    def _iterator(self):
      for text in self.text:
        yield text

    def Preprocess_Text(self):
      for text in self._iterator():
        # Remove numbers from the string
        cleaned_text = re.sub(r'\d+', '', text)

        # Token creation & Lemmatization
        doc = self.nlp(cleaned_text)
        tokens = [token.lemma_ for token in doc]
        tokens = [token.strip().lower() for token in tokens]

        # Stopword & Punctuation Removal
        cleaned_tokens = [token for token in tokens if token not in self.stopwords]
        token_filters = filter(None, [self.special_chars.sub(' ', token) for token in cleaned_tokens])
        new_text = ' '.join(token_filters)

        # Whitespace removal between pre-existing punctuations & stopwords
        new_text = " ".join(new_text.split())
        self.processed_text.append(new_text)
      return self.processed_text


def preprocess(df):
  """To concatenate aspect category, target_term and original sentence, which will be the input for embedding"""

  tp = TextProcessor(df['sentence'])
  df['sentence'] = tp.Preprocess_Text()
  df['inputs'] = df['aspect_cat'] + " " + df['target_term'] + " " + df['sentence']

  return df[['polarity', 'inputs']]



class DataFrameDataset(torch.legacy.data.Dataset):
  '''
  credit: https://gist.github.com/lextoumbourou/8f90313cbc3598ffbabeeaa1741a11c8
  To create a dataloader suitable for pytorch, we need to transform a dataframe into an iterator
  '''

  def __init__(self, df, text_field, label_field, is_test=False):
    fields = [('text', text_field), ('label', label_field)]
    examples = []
    for i, row in df.iterrows():
      label = row.polarity if not is_test else None
      text = row.inputs
      examples.append(data.Example.fromlist([text, label], fields))

    super().__init__(examples, fields)

  @staticmethod
  def sort_key(ex):
    return len(ex.text)

  @classmethod
  def splits(cls, text_field, label_field, df):
    data, data2 = (None, None)

    data = cls(df.copy(), text_field, label_field)

    return tuple(d for d in (data, data2) if d is not None)

# Build the classifier
class CNN(nn.Module):
  def __init__(self,
               vocab_size,
               embedding_dim,
               output_dim,
               n_filters,
               dropout,
               pad_idx):
    super().__init__()

    self.embedding = nn.Embedding(vocab_size, embedding_dim,
                                  padding_idx=pad_idx)

    # uni-gram
    self.conv1 = nn.Conv2d(in_channels=1,
                           out_channels=n_filters,
                           kernel_size=(1, embedding_dim))
    # bi-gram
    self.conv2 = nn.Conv2d(in_channels=1,
                           out_channels=n_filters,
                           kernel_size=(2, embedding_dim))
    # tri-gram
    self.conv3 = nn.Conv2d(in_channels=1,
                           out_channels=n_filters,
                           kernel_size=(3, embedding_dim))
    # quart-gram
    self.conv4 = nn.Conv2d(in_channels=1,
                           out_channels=n_filters,
                           kernel_size=(4, embedding_dim))

    # (# of conv layers * # of filter, output_dim)
    self.linear = nn.Linear(4 * n_filters, output_dim)

    self.dropout = nn.Dropout(dropout)

  def forward(self, text):
    embedded = self.embedding(text)

    embedded = embedded.unsqueeze(1)

    conved1 = F.relu(self.conv1(embedded).squeeze(3))
    pooled1 = F.max_pool1d(conved1, conved1.shape[2]).squeeze(2)

    conved2 = F.relu(self.conv2(embedded).squeeze(3))
    pooled2 = F.max_pool1d(conved2, conved2.shape[2]).squeeze(2)

    conved3 = F.relu(self.conv3(embedded).squeeze(3))
    pooled3 = F.max_pool1d(conved3, conved3.shape[2]).squeeze(2)

    conved4 = F.relu(self.conv4(embedded).squeeze(3))
    pooled4 = F.max_pool1d(conved4, conved4.shape[2]).squeeze(2)

    concatenated = torch.cat([pooled1, pooled2, pooled3, pooled4], dim=1)

    dropout = self.dropout(concatenated)

    output = self.linear(concatenated)

    return output


def accuracy(preds, y):
  """
  Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
  """
  # #round predictions to the closest integer
  top_pred = preds.argmax(1, keepdim=True)
  correct = top_pred.eq(y.view_as(top_pred)).sum()
  acc = correct.float() / y.shape[0]
  return acc

def train_model(model, iterator, optimizer, criterion):
  epoch_loss = 0
  epoch_acc = 0

  model.train()

  for batch in iterator:
    optimizer.zero_grad()

    predictions = model(batch.text).squeeze(1)

    loss = criterion(predictions, batch.label.long())

    acc = accuracy(predictions, batch.label)

    loss.backward()


    optimizer.step()

    epoch_loss += loss.item()
    epoch_acc += acc.item()

  return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate_model(model, iterator, criterion):

  epoch_loss = 0
  epoch_acc = 0

  model.eval()

  with torch.no_grad():
    for batch in iterator:
      predictions = model(batch.text).squeeze(1)

      loss = criterion(predictions, batch.label.long())

      acc = accuracy(predictions, batch.label)

      epoch_loss += loss.item()
      epoch_acc += acc.item()

  return epoch_loss / len(iterator), epoch_acc / len(iterator)

import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
import matplotlib.pyplot as plt

def plot_acc(train_acc, val_acc, nb_epochs):
    plt.plot(list(range(nb_epochs+1))[1:], train_acc)
    plt.plot(list(range(nb_epochs+1))[1:], val_acc)
    plt.legend(['train', 'val'])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()
    # plt.savefig('{}/chart.png'.format(save_folder))


def get_predsevaluate(model, iterator):
  outputs = []
  pred_labels = []

  model.eval()

  with torch.no_grad():

    for batch in iterator:
      output = model(batch.text).squeeze(1)
      outputs.append(output)

    for batch in outputs:
      for sentence in batch:
        pred = sentence.argmax()
        pred_labels.append(pred.item())

  return pred_labels
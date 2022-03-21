
import pandas as pd
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext import data, datasets
from sklearn.model_selection import train_test_split



from Utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# set random seed
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize = 'spacy',
                  tokenizer_language = 'en_core_web_sm',
                  batch_first = True)
LABEL = data.LabelField(dtype = torch.float)
MAX_VOCAB_SIZE = 25000



BATCH_SIZE = 5


N_EPOCHS = 10
global model


class Classifier:
    """The Classifier"""


    #############################################
    def train(self, trainfile, devfile=None):
        """
        Trains the classifier model on the training set stored in file trainfile
        WARNING: DO NOT USE THE DEV DATA AS TRAINING EXAMPLES, YOU CAN USE THEM ONLY FOR THE OPTIMIZATION
         OF MODEL HYPERPARAMETERS
        """
        train_set = load_data(trainfile)

        # train, validation test split
        train, val = train_test_split(train_set, test_size=0.1)
        train_cleaned = preprocess(train)
        val_cleaned = preprocess(val)
        train_ds = DataFrameDataset.splits(
            text_field=TEXT, label_field=LABEL,
            df=train_cleaned)
        train_ds = train_ds[0]

        val_ds = DataFrameDataset.splits(
            text_field=TEXT, label_field=LABEL,
            df=val_cleaned)
        val_ds = val_ds[0]




        TEXT.build_vocab(train_ds,
                         max_size=MAX_VOCAB_SIZE,
                         vectors="glove.6B.100d",
                         unk_init=torch.Tensor.normal_)

        LABEL.build_vocab(train_ds)

        # Build iterator
        train_iterator = data.BucketIterator(
            (train_ds),
            batch_size=BATCH_SIZE,
            sort_key=train_ds.sort_key,
            device=device)

        val_iterator = data.BucketIterator(
            (val_ds),
            batch_size=BATCH_SIZE,
            sort_key=val_ds.sort_key,
            device=device)

        INPUT_DIM = len(TEXT.vocab)
        EMBEDDING_DIM = 100
        N_FILTERS = 200

        OUTPUT_DIM = 3  # 3 labels
        DROPOUT = 0.15
        PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

        model = CNN(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM, N_FILTERS, DROPOUT, PAD_IDX)
        # load pre-trained embeddings

        pretrained_embeddings = TEXT.vocab.vectors
        model.embedding.weight.data.copy_(pretrained_embeddings)

        # zero the initial weights of the unknown and padding tokens

        UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

        model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
        model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
        # Parameters for model training
        LR = 1e-4
        optimizer = optim.Adam(model.parameters(), lr=LR)

        criterion = nn.CrossEntropyLoss()

        model = model.to(device)
        criterion = criterion.to(device)

        best_valid_loss = float('inf')
        total_train_acc = []
        total_val_acc = []

        for epoch in range(N_EPOCHS):

            start_time = time.time()

            train_loss, train_acc = train_model(model, train_iterator, optimizer, criterion)
            valid_loss, valid_acc = evaluate_model(model, val_iterator, criterion)

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            total_train_acc.append(train_acc)
            total_val_acc.append(valid_acc)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), 'model.pt')

            print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

        # plot_acc(total_train_acc, total_val_acc, N_EPOCHS)


    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        test_cleaned = preprocess(load_data(datafile))
        test_ds = DataFrameDataset.splits(
            text_field=TEXT, label_field=LABEL,
            df=test_cleaned)
        test_ds = test_ds[0]

        # Build iterator
        test_iterator = data.BucketIterator(
            (test_ds),
            batch_size=BATCH_SIZE,
            sort_key=test_ds.sort_key,
            device=device)

        # model = CNN(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM, N_FILTERS, DROPOUT, PAD_IDX)
        model.load_state_dict(torch.load('model.pt'))

        test_loss, test_acc = evaluate_model(model, test_iterator, criterion)

        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

        return get_predsevaluate(model, test_iterator)




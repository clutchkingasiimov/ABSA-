import pandas as pd
import spacy
from tqdm import tqdm
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from torch.utils.data import Dataset, TensorDataset, RandomSampler, SequentialSampler, DataLoader
from transformers import BertTokenizer
# tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

from Utils import _create_examples
from Utils import *
tokenizer = AutoTokenizer.from_pretrained("activebus/BERT-XD_Review")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Classifier:
    """The Classifier"""

    def __init__(self,batch_size = 12, epochs=5, max_seq_length = 138, learning_rate = 3e-5,warmup_proportion=0.1):
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_seq_length = max_seq_length
        self.learning_rate = learning_rate
        self.warmup_proportion = warmup_proportion
        self.model = None

    def train(self, trainfile, devfile=None):
        train_set = pd.read_csv(trainfile, sep='\t',
                                names=['polarity', 'aspect', 'target', 'offset', 'sentence'])
        label_list = ["positive", "negative", "neutral"]
        train_examples = _create_examples(train_set, 'train')
        num_train_steps = int(len(train_examples) / self.batch_size) * self.epochs

        train_features = convert_examples_to_features(
            train_examples, label_list, self.max_seq_length, tokenizer)
        #     logger.info("***** Running training *****")
        #     logger.info("  Num examples = %d", len(train_examples))
        #     logger.info("  Batch size = %d", args.train_batch_size)
        #     logger.info("  Num steps = %d", num_train_steps)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)

        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)

        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_ids)

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_list))
        self.model.to(device)
        # Prepare optimizer
        param_optimizer = [(k, v) for k, v in self.model.named_parameters() if v.requires_grad == True]
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        t_total = num_train_steps
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=self.learning_rate,
                             warmup=self.warmup_proportion,
                             t_total=t_total)

        global_step = 0
        self.model.train()
        for _ in range(self.epochs):
            for step, batch in enumerate(tqdm(train_dataloader)):
                batch = tuple(t.to(device) for t in batch)
                input_ids, segment_ids, input_mask, label_ids = batch
                loss = self.model(input_ids, segment_ids, input_mask, label_ids)
                loss.backward()

                lr_this_step = self.learning_rate * warmup_linear(global_step / t_total, self.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

        torch.save(self.model, "model.pt")

    def predict(self, datafile,max_seq_length=128, batch_size=6):
        label_list = ["positive", "negative", "neutral"]
        test_set = pd.read_csv(datafile, sep='\t',
                                names=['polarity', 'aspect', 'target', 'offset', 'sentence'])
        eval_examples = _create_examples(test_set, 'test')

        #     tokenizer = BertTokenizer.from_pretrained(modelconfig.MODEL_ARCHIVE_MAP[args.bert_model])

        eval_features = convert_examples_to_features(eval_examples, label_list, max_seq_length, tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

        model = torch.load("model.pt")
        model.to(device)
        model.eval()

        full_logits = []
        predicted_labels = []
        full_label_ids = []
        for step, batch in enumerate(tqdm(eval_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            #         batch = tuple(t.cuda() for t in batch)
            input_ids, segment_ids, input_mask, label_ids = batch

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.cpu().numpy()

            full_logits.extend(logits.tolist())
            full_label_ids.extend(label_ids.tolist())
            predicted_labels = np.argmax(full_logits, axis=1)
            label_dict = {0: 'positive', 1: 'negative', 2: 'neutral'}
            slabels = [label_dict[i] for i in predicted_labels]

        return slabels




import random
import os
from torch.utils.data import Dataset
import torch
import csv
from toolz.sandbox import unzip
from cytoolz import concat
import json
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import pickle
from tqdm import tqdm
import jsonlines


class LangIden_Dataset_train(Dataset):
    def __init__(self, tokenizer, file_path, label2id, max_seq_length, is_train=True):
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.max_length = max_seq_length
        self.cls = self.tokenizer.cls_token
        self.sep = self.tokenizer.sep_token
        self.label2id = label2id
        self.examples, self.label2index, self.max_len = self.read_example(file_path)

    def read_example(self, path):
        data_list = []
        with open(path, 'r') as csv_file:
            all_lines = csv.reader(csv_file)
            for one_line in all_lines:
                data_list.append(one_line)
        data_list = data_list[1:]
        pbar = tqdm(total=len(data_list))
        examples = []
        label2index = {k: [] for k in self.label2id.keys()}
        for index, item in enumerate(data_list):
            label = item[0]
            text = item[1]
            tokenized_text = self.tokenizer(text)
            ids = tokenized_text['input_ids']
            if len(ids) > self.max_length:
                ids = ids[:self.max_length - 1]
                ids += [self.tokenizer.sep_token_id]
            attn_mask = [1] * len(ids)
            label2index[label].append(index)
            examples.append({
                'label': label,
                'labelID': torch.tensor(self.label2id[label]),
                'ids': torch.tensor(ids),
                'attn_mask': torch.tensor(attn_mask)
            })
            pbar.update(1)
        # for balance training
        for key in label2index.keys():
            random.shuffle(label2index[key])
        max_len = max([len(label2index[k]) for k in label2index.keys()])
        return examples, label2index, max_len

    def __len__(self):
        return self.max_len

    def __getitem__(self, i):
        res = []
        for key in self.label2index.keys():
            index = self.label2index[key][i]
            input = self.examples[index]
            input_ids = input['ids']
            input_mask = input['attn_mask']
            label = input['labelID']
            res.append((index, input_ids, input_mask, label))

        return res

    def data_collate(self, inputs):
        (index, input_ids, input_mask, label) = map(list, unzip(concat(inputs)))

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        input_mask = pad_sequence(input_mask, batch_first=True, padding_value=0)
        label = torch.stack(label)
        batch = {'index': index, 'input_ids': input_ids, 'input_mask': input_mask,
                 'label': label}

        return batch


class LangIden_Dataset_eval(Dataset):
    def __init__(self, tokenizer, file_path, label2id, max_seq_length, is_train=True):
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.max_length = max_seq_length
        self.cls = self.tokenizer.cls_token
        self.sep = self.tokenizer.sep_token
        self.label2id = label2id
        self.examples = self.read_example(file_path)

    def read_example(self, path):
        data_list = []
        with open(path, 'r') as csv_file:
            all_lines = csv.reader(csv_file)
            for one_line in all_lines:
                data_list.append(one_line)
        data_list = data_list[1:]
        pbar = tqdm(total=len(data_list))
        examples = []
        for index, item in enumerate(data_list):
            label = item[0]
            text = item[1]
            tokenized_text = self.tokenizer(text)
            ids = tokenized_text['input_ids']
            if len(ids) > self.max_length:
                ids = ids[:self.max_length - 1]
                ids += [self.tokenizer.sep_token_id]
            attn_mask = [1] * len(ids)
            examples.append({
                'label': label,
                'labelID': torch.tensor(self.label2id[label]),
                'ids': torch.tensor(ids),
                'attn_mask': torch.tensor(attn_mask)
            })
            pbar.update(1)
        # for balance training
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        input = self.examples[i]
        input_ids = input['ids']
        input_mask = input['attn_mask']
        label = input['label']

        return [(i, input_ids, input_mask, label)]

    def data_collate(self, inputs):
        (index, input_ids, input_mask, label) = map(list, unzip(concat(inputs)))

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        input_mask = pad_sequence(input_mask, batch_first=True, padding_value=0)

        batch = {'index': index, 'input_ids': input_ids, 'input_mask': input_mask,
                 'label': label}

        return batch


class LangIden_Dataset_XNLI_eval(Dataset):
    def __init__(self, tokenizer, file_path, label2id, max_seq_length, is_train=True):
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.max_length = max_seq_length
        self.cls = self.tokenizer.cls_token
        self.sep = self.tokenizer.sep_token
        self.label2id = label2id
        self.examples = self.read_example(file_path)

    def read_example(self, path):
        data_list = []
        with open(path, 'r') as csv_file:
            all_lines = csv.reader(csv_file,delimiter='\t')
            for one_line in all_lines:
                data_list.append(one_line)
        pbar = tqdm(total=len(data_list))
        examples = []
        label_index = data_list[0]
        data_list = data_list[1:]
        for item_list in data_list:
            for index, label in enumerate(label_index):
                text = item_list[index]
                tokenized_text = self.tokenizer(text)
                ids = tokenized_text['input_ids']
                if len(ids) > self.max_length:
                    ids = ids[:self.max_length - 1]
                    ids += [self.tokenizer.sep_token_id]
                attn_mask = [1] * len(ids)
                examples.append({
                    'label': label,
                    'labelID': torch.tensor(self.label2id[label]),
                    'ids': torch.tensor(ids),
                    'attn_mask': torch.tensor(attn_mask)
                })
            pbar.update(1)
        # for balance training
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        input = self.examples[i]
        input_ids = input['ids']
        input_mask = input['attn_mask']
        label = input['label']

        return [(i, input_ids, input_mask, label)]

    def data_collate(self, inputs):
        (index, input_ids, input_mask, label) = map(list, unzip(concat(inputs)))

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        input_mask = pad_sequence(input_mask, batch_first=True, padding_value=0)

        batch = {'index': index, 'input_ids': input_ids, 'input_mask': input_mask,
                 'label': label}

        return batch

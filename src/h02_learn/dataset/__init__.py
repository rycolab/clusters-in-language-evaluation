import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer

from util import constants
from util import util
from .alphabet import Alphabet
from .sentences import SentenceDataset


def generate_batch(batch):
    # return batch
    tensor = batch[0]
    batch_size = len(batch)
    max_length = max(len(entry) for entry in batch) - 1  # Does not need to predict SOS

    x = tensor.new_zeros(batch_size, max_length)
    y = tensor.new_zeros(batch_size, max_length)

    for i, item in enumerate(batch):
        word = item
        word_len = len(word) - 1  # Does not need to predict SOS
        x[i, :word_len] = word[:-1]
        y[i, :word_len] = word[1:]

    x, y = x.to(device=constants.device), y.to(device=constants.device)
    return x, y


def load_data(fname):
    return util.read_data(fname)


def tokenize_data(data):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    data = [tokenizer.encode(sentence, max_length=512, truncation=True) for sentence in data]
    return data, tokenizer


def split_data(data, train_ratio):
    data_size = len(data)
    train_size = int(train_ratio * data_size)
    return data[:train_size], data[train_size:]


def get_data_loader(
        data, batch_size, shuffle):
    trainset = SentenceDataset(data)
    return DataLoader(trainset, batch_size=batch_size,
                      shuffle=shuffle, collate_fn=generate_batch)


def get_data_loaders(
        data, train_ratio, batch_size, eval_batch_size):
    data, tokenizer = tokenize_data(data)
    splits = split_data(data, train_ratio)

    trainloader = get_data_loader(
        splits[0], batch_size=batch_size, shuffle=True)
    devloader = get_data_loader(
        splits[1], batch_size=eval_batch_size, shuffle=True)
    return trainloader, devloader, tokenizer

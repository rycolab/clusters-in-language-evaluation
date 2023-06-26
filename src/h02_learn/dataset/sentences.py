import torch
from torch.utils.data import Dataset


class SentenceDataset(Dataset):
    # pylint: disable=no-member

    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sentence = self.sentences[index]
        return torch.LongTensor(sentence)

import torch
from torch import nn
import torch.nn.functional as F

from util import constants
from .base import BaseLM


class LstmLM(BaseLM):
    # pylint: disable=arguments-differ,not-callable
    name = 'lstm'

    def __init__(self, alphabet_size, embedding_size=32, hidden_size=128,
                 nlayers=2, dropout=.3, ignore_index=-1):
        super().__init__(alphabet_size, embedding_size, hidden_size,
                         nlayers, dropout, ignore_index)

        self.embedding = nn.Embedding(alphabet_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, nlayers,
                            dropout=(dropout if nlayers > 1 else 0),
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, embedding_size)
        self.out = nn.Linear(embedding_size, alphabet_size)

        # Tie weights
        self.out.weight = self.embedding.weight

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.ignore_index, reduction='none') \
            .to(device=constants.device)

    def forward(self, x):
        x_emb = self.dropout(self.embedding(x))

        c_t, _ = self.lstm(x_emb)
        c_t = self.dropout(c_t).contiguous()

        hidden = F.relu(self.linear(c_t))
        logits = self.out(hidden)
        return logits

    def get_loss(self, logits, y):
        return self.criterion(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1)) \
            .reshape_as(y)

    def get_word_log_probability(self, x, y):
        logits = self(x)
        logprobs = self.get_loss(logits, y).sum(-1)
        return -logprobs

    def sample(self, alphabet, n_samples, temperature=.5):
        eos = alphabet.EOS_IDX
        pad = alphabet.PAD_IDX
        x = torch.LongTensor(
            [[alphabet.SOS_IDX] for _ in range(n_samples)]) \
            .to(device=constants.device)

        first = True
        ended = torch.zeros(
            x.shape[0], dtype=torch.bool).to(device=constants.device)

        while True:
            logits = self(x) / temperature
            logits = self.mask_logits(logits, first, ended, eos, pad)
            probs = F.softmax(logits, dim=-1)

            first = False

            y = probs.multinomial(1)
            x = torch.cat([x, y], dim=-1)

            y = y.squeeze(-1)
            ended = ended | (y == eos) | (y == pad)

            if ended.all():
                break

        samples = [alphabet.idx2word(item[1:-1].cpu().numpy()) for item in x]
        return [''.join(item) for item in samples]

    @staticmethod
    def mask_logits(logits, first, ended, eos, pad):
        logits = logits[:, -1, :]
        if first:
            logits[:, eos] = -float('inf')
            logits[:, pad] = -float('inf')
        else:
            logits[ended, pad] += 100000000
            logits[~ended, pad] -= float('inf')

        return logits

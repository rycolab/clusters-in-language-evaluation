import torch
from torch import optim
import tqdm

from util import constants
from util.train_info import TrainInfo
from .lstm import LstmLM as BaseLstmLM


class LstmLM(BaseLstmLM):
    # pylint: disable=arguments-differ,not-callable
    name = 'lstm'

    def __init__(self, *args, eval_batches=100, wait_iterations=5, **kwargs):
        super().__init__(*args, **kwargs)

        self.optimizer = optim.AdamW(self.parameters())
        self.eval_batches = eval_batches
        self.wait_iterations = wait_iterations * eval_batches

    def learn(self, trainloader, devloader):
        train_info = TrainInfo(self.wait_iterations, self.eval_batches)

        while not train_info.finish:
            for x, y in tqdm.tqdm(trainloader, total=len(trainloader)):
                loss = self.train_batch(x, y)
                train_info.new_batch(loss)

                if train_info.eval:
                    dev_loss = self.evaluate(devloader, max_samples=1000)

                    if train_info.is_best(dev_loss):
                        self.set_best()
                    elif train_info.finish:
                        break

                    train_info.print_progress(dev_loss)

        self.recover_best()
        return loss, dev_loss

    def train_batch(self, x, y, by_character=False):
        self.optimizer.zero_grad()
        y_hat = self(x)
        loss = self.get_loss(y_hat, y).sum(-1)
        if by_character:
            word_lengths = (y != 0).sum(-1)
            loss = (loss / word_lengths).mean()
        else:
            loss = loss.mean()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, evalloader, max_samples=None):
        self.eval()
        with torch.no_grad():
            result = self._evaluate(evalloader, max_samples=max_samples)
        self.train()
        return result

    def _evaluate(self, evalloader, max_samples=None):
        dev_loss, n_instances = 0, 0
        for x, y in evalloader:
            y_hat = self(x)
            loss = self.get_loss(y_hat, y).sum(-1)
            dev_loss += loss.sum()
            n_instances += loss.shape[0]

            if max_samples is not None and n_instances >= max_samples:
                break

        return (dev_loss / n_instances).item()

    def score_instance(self, sentence):
        self.eval()
        with torch.no_grad():
            sentence = torch.LongTensor([sentence]).to(device=constants.device)
            x, y = sentence[:, :-1], sentence[:, 1:]

            y_hat = self(x)
            loss = self.get_loss(y_hat, y).sum(-1)

        return loss.item()

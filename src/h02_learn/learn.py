import os
import sys
import argparse

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from h02_learn.dataset import get_data_loaders
from h02_learn.model import Categorical
from h02_learn.model import LstmLM
from util import constants
from util import util


def get_args():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--data-fpath", type=str, required=True)
    parser.add_argument('--data-type', type=str, required=True)
    # Model
    parser.add_argument("--num-buckets", type=int, default=500)
    parser.add_argument("--laplace-smoothing", type=int, default=1)
    # Others
    parser.add_argument("--model-fpath", type=str, required=True)
    parser.add_argument("--seed", type=int, default=20)

    args = parser.parse_args()
    print(args)
    return args


def train_categorical(data, num_buckets, laplace_smoothing):
    clusters = data['clusters']

    model = Categorical(num_buckets, laplace_smoothing=laplace_smoothing)
    model.train(clusters)
    return model


def train_text(data):
    sentences = data['texts']
    trainloader, devloader, tokenizer = get_data_loaders(
        sentences, train_ratio=.8, batch_size=16, eval_batch_size=16)

    model = LstmLM(tokenizer.vocab_size).to(device=constants.device)
    model.learn(trainloader, devloader)
    return model


def save_checkpoints(model, model_fpath):
    util.write_data(model_fpath, model)


def train_model(data_fpath, data_type, model_fpath, args):
    data = util.read_data(data_fpath)

    if data_type == 'clusters':
        model = train_categorical(
            data, num_buckets=args.num_buckets, laplace_smoothing=args.laplace_smoothing)
        save_checkpoints(model, model_fpath)
    elif data_type == 'text':
        model = train_text(data)
        model.save(model_fpath)
    else:
        raise ValueError(f'Learn should not be run with option {data_type}')



def main():
    args = get_args()

    train_model(args.data_fpath, args.data_type, args.model_fpath, args)


if __name__ == '__main__':
    main()

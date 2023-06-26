import os
import sys
import argparse
import numpy as np
import kenlm

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from h02_learn.dataset import tokenize_data
from h02_learn.model import LstmLM
from util import util


def get_args():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--model-data-fpath", type=str, required=True)
    parser.add_argument("--human-data-fpath", type=str, required=True)
    parser.add_argument('--data-type', type=str, required=True)
    parser.add_argument('--model-type', type=str, required=True,
                        choices=['ngram', 'categorical', 'lstm'])
    # Model
    parser.add_argument("--num-buckets", type=int, default=500)
    # Others
    parser.add_argument("--model-fpath", type=str, required=True)
    parser.add_argument("--surprisals-fpath", type=str, required=True)

    args = parser.parse_args()
    print(args)
    return args


def run_ngram(model, data):
    texts = data['texts']

    surprisals = np.array([
        - model.score(sentence) for sentence in texts
    ])
    return surprisals


def run_lstm(model, data):
    texts = data['texts']

    sentences, _ = tokenize_data(texts)
    surprisals = np.array([
        model.score_instance(sentence) for sentence in sentences
    ])
    return surprisals


def run_categorical(model, data):
    clusters = data['clusters']

    surprisals = np.array([
        - model.score_instance(label) for label in clusters
    ])
    return surprisals


def load_ngram(model_fpath):
    model = kenlm.Model(model_fpath)
    return model


def load_checkpoints(model_fpath):
    model = util.read_data(model_fpath)
    return model


def save_surprisals(ids_model, surprisals_model, ids_human, surprisals_human, surprisals_fpath):
    save_data = {
        'model_text': {
            'ids': ids_model,
            'surprisals': surprisals_model,
        },
        'human_text': {
            'ids': ids_human,
            'surprisals': surprisals_human,
        }
    }
    util.write_data(surprisals_fpath, save_data)


def get_text_surprisal(data_type, model_type, data_fpath, model_fpath):
    data = util.read_data(data_fpath)

    if data_type == 'text' and model_type == 'ngram':
        model = load_ngram(model_fpath)
        surprisals = run_ngram(model, data)
    elif data_type == 'text' and model_type == 'lstm':
        model = LstmLM.load(model_fpath)
        surprisals = run_lstm(model, data)
    elif data_type == 'clusters' and model_type == 'categorical':
        model = load_checkpoints(model_fpath)
        surprisals = run_categorical(model, data)

    return data['ids'], surprisals


def get_surprisals(data_type, model_type, model_data_fpath, human_data_fpath,
                   model_fpath, surprisals_fpath):
    ids_human, surprisals_human = get_text_surprisal(data_type, model_type, human_data_fpath,
                                                     model_fpath)
    ids_model, surprisals_model = get_text_surprisal(data_type, model_type, model_data_fpath,
                                                     model_fpath)

    save_surprisals(ids_model, surprisals_model, ids_human, surprisals_human, surprisals_fpath)


def main():
    args = get_args()

    get_surprisals(args.data_type, args.model_type, args.model_data_fpath, args.human_data_fpath,
                   args.model_fpath, args.surprisals_fpath)


if __name__ == '__main__':
    main()

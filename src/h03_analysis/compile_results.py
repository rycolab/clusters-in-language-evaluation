import os
# from os import listdir
# from os.path import isdir, join
import sys
from types import SimpleNamespace
import argparse
import pandas as pd

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from util import util


def get_args():
    parser = argparse.ArgumentParser()
    # Data type
    parser.add_argument("--dataset", type=str, required=True)
    # Results
    parser.add_argument("--checkpoints-dir", type=str, required=True)
    parser.add_argument("--results-fpath", type=str, required=True)

    args = parser.parse_args()
    print(args)
    return args


def list_subdirs(directory):
    return [
        f for f in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, f))
    ]


def load_scores(scores_fpath, key='score'):
    data = util.read_data(scores_fpath)
    return data[key]


def get_representation_results(auc_scores_fpath, jensen_scores_fpath, info):

    if (not os.path.isfile(auc_scores_fpath)) or \
            (not os.path.isfile(jensen_scores_fpath)):
        return None

    auc_score = load_scores(auc_scores_fpath)
    js_score = load_scores(jensen_scores_fpath, key='js_score')
    forward_score = load_scores(jensen_scores_fpath, key='forward_score')
    backward_score = load_scores(jensen_scores_fpath, key='backward_score')
    exponentiated_score = load_scores(jensen_scores_fpath, key='exponentiated_score')

    return {
        'dataset': info.dataset,
        'model': info.model,
        'estimator': info.estimator,
        'estimator_type': info.estimator_type,
        'seed': info.seed,

        'auc_score': auc_score,
        'js_score': js_score,
        'forward_score': forward_score,
        'backward_score': backward_score,
        'exponentiated_score': exponentiated_score,
    }


def get_model_results(checkpoints_dir, dataset, seed, model):
    model_dir = os.path.join(checkpoints_dir, seed, model)
    results = []
    estimators = [
        ('bert-base-cased/average', 'cluster'),
        ('bert-large-cased/average', 'cluster'),
        ('gpt2-xl/final', 'cluster'),
        ('gpt2-large/final', 'cluster'),
        ('gpt2-medium/final', 'cluster'),
        ('gpt2/final', 'cluster'),
        ('bert-base-cased/final', 'cluster'),
        ('bert-large-cased/final', 'cluster'),
        ('gpt2-xl/average', 'cluster'),
        ('gpt2-large/average', 'cluster'),
        ('gpt2-medium/average', 'cluster'),
        ('gpt2/average', 'cluster'),
        ('ngram', 'text'),
        ('lstm', 'text')
    ]

    for estimator, estimator_type in estimators:
        estimator_dir = os.path.join(model_dir, estimator)

        auc_scores_fpath = os.path.join(
            estimator_dir, f'empirical_scores.{estimator_type}.pickle')
        jensen_scores_fpath = os.path.join(
            estimator_dir, f'divergences_scores.{estimator_type}.pickle')

        info = SimpleNamespace(
            dataset=dataset, model=model, estimator=estimator,
            estimator_type=estimator_type, seed=seed)
        result = get_representation_results(
            auc_scores_fpath, jensen_scores_fpath, info)

        if result is not None:
            results += [result]

    return results


def main():
    args = get_args()

    checkpoints_dir = os.path.join(args.checkpoints_dir, args.dataset)

    results = []
    for seed in list_subdirs(checkpoints_dir):
        seed_dir = os.path.join(checkpoints_dir, seed)
        for model in list_subdirs(seed_dir):
            results_model = get_model_results(checkpoints_dir, args.dataset, seed, model)
            results += results_model

    df = pd.DataFrame(results)
    df.to_csv(args.results_fpath, sep='\t')

    print(df)


if __name__ == '__main__':
    main()

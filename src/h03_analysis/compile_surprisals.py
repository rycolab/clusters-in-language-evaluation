import os
import sys
from types import SimpleNamespace
import argparse
import pandas as pd

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from h03_analysis.compile_results import list_subdirs


def get_args():
    parser = argparse.ArgumentParser()
    # Data type
    parser.add_argument("--dataset", type=str, required=True)
    # Results
    parser.add_argument("--checkpoints-dir", type=str, required=True)
    parser.add_argument("--src-fname", type=str, required=True)
    # Save path
    parser.add_argument("--results-fpath", type=str, required=True)

    args = parser.parse_args()
    print(args)
    return args


def load_scores(corr_fpath):
    df = pd.read_csv(corr_fpath, sep='\t', index_col=0)
    return df


def get_representation_results(corr_fpath, info):
    if not os.path.isfile(corr_fpath):
        return None

    df = load_scores(corr_fpath)
    # text_corr = df[df.index == 'text'].corr.item()
    text_corr = df[df.index == 'text']['corr'].item()
    text_pvalue = df[df.index == 'text']['pvalue'].item()
    cluster_corr = df[df.index == 'cluster']['corr'].item()
    cluster_pvalue = df[df.index == 'cluster']['pvalue'].item()

    return {
        'dataset': info.dataset,
        'model': info.model,
        'representation': info.representation,
        'representation_type': info.representation_type,
        'seed': info.seed,
        'text_corr': text_corr,
        'text_pvalue': text_pvalue,
        'cluster_corr': cluster_corr,
        'cluster_pvalue': cluster_pvalue,
    }


def get_model_results(checkpoints_dir, dataset, seed, model, src_fname):
    model_dir = os.path.join(checkpoints_dir, seed, model)

    results = []
    for representation in list_subdirs(model_dir):
        representation_dir = os.path.join(model_dir, representation)

        for representation_type in list_subdirs(representation_dir):

            corr_fpath = os.path.join(
                representation_dir, representation_type, src_fname)

            info = SimpleNamespace(
                dataset=dataset, model=model, representation=representation,
                representation_type=representation_type, seed=seed)
            result = get_representation_results(
                corr_fpath, info)
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
            results_model = get_model_results(checkpoints_dir, args.dataset, seed,
                                              model, args.src_fname)
            results += results_model

    df = pd.DataFrame(results)
    df.to_csv(args.results_fpath, sep='\t')

    print(df)


if __name__ == '__main__':
    main()

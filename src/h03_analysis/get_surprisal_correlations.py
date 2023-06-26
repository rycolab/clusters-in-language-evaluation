import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy import stats
# from nltk.util import ngrams

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from h01_data.get_surprisals import load_surprisals as load_orig_surprisals
from util import util


def get_args():
    parser = argparse.ArgumentParser()
    # Results
    parser.add_argument("--surprisals-orig-fpath", type=str, required=True)
    parser.add_argument("--surprisals-cluster-fpath", type=str, required=True)
    parser.add_argument("--surprisals-text-fpath", type=str, required=True)
    # Save
    parser.add_argument("--results-fpath", type=str, required=True)

    args = parser.parse_args()
    print(args)
    return args


def load_surprisals(surprisals_fpath):
    data = util.read_data(surprisals_fpath)
    data = data['model_text']
    return np.array(data['ids']), data['surprisals']


def load_all_surprisals(surprisals_orig_fpath, surprisals_cluster_fpath, surprisals_text_fpath):
    ids_orig, surprisals_orig = load_orig_surprisals(surprisals_orig_fpath)
    ids_cluster, surprisals_cluster = load_surprisals(surprisals_cluster_fpath)
    ids_text, surprisals_text = load_surprisals(surprisals_text_fpath)

    assert (ids_orig == ids_cluster).all()
    assert (ids_orig == ids_text).all()

    return surprisals_orig, surprisals_cluster, surprisals_text


def get_correlations(surprisals_orig_fpath, surprisals_cluster_fpath, surprisals_text_fpath):
    surprisals_orig, surprisals_cluster, surprisals_text = \
        load_all_surprisals(surprisals_orig_fpath, surprisals_cluster_fpath, surprisals_text_fpath)

    corr_cluster, pvalue_cluster = stats.spearmanr(surprisals_orig, surprisals_cluster)
    corr_text, pvalue_text = stats.spearmanr(surprisals_orig, surprisals_text)

    results = {
        'text': [corr_text, pvalue_text],
        'cluster': [corr_cluster, pvalue_cluster],
    }
    df = pd.DataFrame.from_dict(results, columns=['corr', 'pvalue'], orient='index')
    print(df)
    return df


def main():
    args = get_args()

    df = get_correlations(args.surprisals_orig_fpath, args.surprisals_cluster_fpath,
                          args.surprisals_text_fpath)
    df.to_csv(args.results_fpath, sep='\t')


if __name__ == '__main__':
    main()

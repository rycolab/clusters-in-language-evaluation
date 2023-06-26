import os
import sys
import itertools
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path.append('./src/')
from util import constants
from util import util

aspect = {
    'height': 3.5,
    'font_scale': 1.5,
    'labels': True,
    'name_suffix': '',
    'ratio': 3.25,
}
sns.set_palette("Set2")
sns.set_context("notebook", font_scale=aspect['font_scale'])
mpl.rc('font', family='serif', serif='Times New Roman')


def get_args():
    parser = argparse.ArgumentParser()
    # Save
    parser.add_argument('--correlations-fpath', type=str, required=True)
    parser.add_argument("--results-fpath", type=str, required=True)

    args = parser.parse_args()
    print(args)
    return args


def str_for_table(corr, pvalue):
    if pvalue < 0.01:
        pvalue_str = '$\\ddagger$'
    elif pvalue < 0.05:
        pvalue_str = '$\\dagger$'
    else:
        pvalue_str = ''
    return '%.2f%s' % (corr, pvalue_str)


def melt_human_judgements(df, human_score_name, plot_name, is_original=True):
    dfs = []

    model_score_names = ['auc_score', 'js_score', 'forward_score',
                         'backward_score', 'exponentiated_score']
    for model_score_name in model_score_names:
        df[model_score_name] = df['%s-%s' % (model_score_name, human_score_name)]

    # estimator_names = {
    #     'bert-large-cased/average': r'$\Delta(p_{c}, q_{c})$' "\n" r'BERT $\mathtt{large}$',
    #     'bert-base-cased/average': r'$\Delta(p_{c}, q_{c})$' "\n" r'BERT $\mathtt{base}$',
    #     'gpt2-xl/final': r'$\Delta(p_{c}, q_{c})$' "\n" r'GPT-2 $\mathtt{xl}$',
    #     'gpt2-large/final': r'$\Delta(p_{c}, q_{c})$' "\n" r'GPT-2 $\mathtt{large}$',
    #     'gpt2-medium/final': r'$\Delta(p_{c}, q_{c})$' "\n" r'GPT-2 $\mathtt{medium}$',
    #     'gpt2/final': r'$\Delta(p_{c}, q_{c})$' "\n" r'GPT-2 $\mathtt{small}$',
    #     'bert-large-cased/final': r'$\Delta(p_{c}, q_{c})$' "\n" r'BERT $\mathtt{large}$',
    #     'bert-base-cased/final': r'$\Delta(p_{c}, q_{c})$' "\n" r'BERT $\mathtt{base}$',
    #     'gpt2-xl/average': r'$\Delta(p_{c}, q_{c})$' "\n" r'GPT-2 $\mathtt{xl}$',
    #     'gpt2-large/average': r'$\Delta(p_{c}, q_{c})$' "\n" r'GPT-2 $\mathtt{large}$',
    #     'gpt2-medium/average': r'$\Delta(p_{c}, q_{c})$' "\n" r'GPT-2 $\mathtt{medium}$',
    #     'gpt2/average': r'$\Delta(p_{c}, q_{c})$' "\n" r'GPT-2 $\mathtt{small}$',
    #     'ngram': r'$\Delta(p_{\mathrm{w}}, q_{\mathrm{w}})$' "\n" r'$n$-gram',
    #     'lstm': r'$\Delta(p_{\mathrm{w}}, q_{\mathrm{w}})$' "\n" r'LSTM',
    # }
    estimator_names = {
        'gpt2/final': r'$\Delta(p_{c}, q_{c})$',
        'gpt2-xl/final': r'$\Delta(p_{c}, q_{c})$',
        'ngram': r'$\Delta(p_{\mathrm{w}}, q_{\mathrm{w}})$',
    }

    estimator_order = {
        'bert-large-cased/average': 4,
        'bert-base-cased/average': 3.5,
        'gpt2-xl/final': 3,
        'gpt2-large/final': 2,
        'gpt2-medium/final': 1,
        'gpt2/final': 0,
        'bert-large-cased/final': 4,
        'bert-base-cased/final': 3.5,
        'gpt2-xl/average': 3,
        'gpt2-large/average': 2,
        'gpt2-medium/average': 1,
        'gpt2/average': 0,
        'ngram': 5,
        'lstm': 6,
    }
    df['estimator_order'] = df['estimator'].apply(lambda x: estimator_order[x])
    df['estimator'] = df['estimator'].apply(lambda x: estimator_names[x])

    model_legend_names = {
        'auc_score': '$\mathtt{AUC}}$',
        'js_score': '$\mathtt{JS}$',
        'forward_score': r'$\rightarrow$',
        'backward_score': r'$\leftarrow$',
        'exponentiated_score': '$\mathtt{exp}$',
    }
    df = pd.melt(df, id_vars=['estimator', 'seed', 'estimator_type', 'estimator_order'], var_name='model_score', value_name='Correlation',
                 value_vars=model_score_names)
    df['model_score_names'] = df['model_score'].apply(lambda x: model_legend_names[x])
    df.sort_values(['model_score_names', 'estimator_order'], inplace=True)

    return df


def plot_correlations(df, human_score_name, results_fpath, plot_name):
    df = df.copy()

    model_order = {
        'auc_score': 4,
        'js_score': 3,
        'forward_score': 1,
        'backward_score': 2,
        'exponentiated_score': 0,
    }

    df = melt_human_judgements(df, human_score_name, plot_name)
    df['model_order'] = df['model_score'].apply(lambda x: model_order[x])
    df.sort_values(['model_order', 'estimator_order'], inplace=True)

    df['Estimator'] = df['estimator']
    df['Correlation (%)'] = df['Correlation'] * 100
    df['Method'] = df['estimator_type']
    df['Score'] = df['model_score_names']

    # Create tiny error bar for constant values
    df['Correlation (%)'] += np.random.randn(df.shape[0]) * .001

    # fig = plt.figure(figsize=(6.4, 2.8))
    if plot_name in ['full_final', 'full_avg']:
        fig = plt.figure(figsize=(18.2, 2.8))
    else:
        fig = plt.figure(figsize=(5.4, 2.8))

    ax = sns.barplot(hue="Score", y="Correlation (%)", x="Estimator", data=df)

    plt.xlabel('')
    plt.ylim([0, 100])
    plt.legend('', frameon=False)

    fname = os.path.join(results_fpath, 'plot_human_judgements--%s--%s.pdf' % (plot_name, human_score_name))
    fig.savefig(fname, bbox_inches='tight')


def load_and_plot_correlations(correlations_fpath, results_fpath, plot_name, estimators=None):
    df = pd.read_csv(correlations_fpath, sep='\t', index_col=0)
    if estimators:
        df = df[df.estimator.isin(estimators)]
    df = df[df.seed.apply(lambda x: int(x[5:])) < 5]

    human_score_names = ['bt_interesting', 'bt_sensible', 'bt_human-like']
    for human_score_name in human_score_names:
        plot_correlations(df, human_score_name, results_fpath, plot_name)


def main():
    args = get_args()

    # load_and_plot_correlations(args.correlations_fpath, args.results_fpath, args.plot_name)
    load_and_plot_correlations(args.correlations_fpath, args.results_fpath, 'main', estimators=['gpt2/final', 'ngram'])


if __name__ == '__main__':
    main()

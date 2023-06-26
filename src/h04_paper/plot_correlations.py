import os
import sys
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path.append('./src/')
from util import constants
from util import util

aspect = {
    'height': 7,
    'font_scale': 1.5,
    'labels': True,
    'name_suffix': '',
    'ratio': 1.625,
}
sns.set_palette("Set2")
sns.set_context("notebook", font_scale=aspect['font_scale'])
mpl.rc('font', family='serif', serif='Times New Roman')


def get_args():
    parser = argparse.ArgumentParser()
    # Save
    parser.add_argument('--correlations-fpath', type=str, required=True)
    parser.add_argument("--results-fname", type=str, required=True)

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


def plot_correlations(df, results_fname):
    df = df.copy()

    legend_names = {
        'text_corr': '$\widehat{q}_{\mathbf{w}}$',
        'cluster_corr': '$\widehat{q}_{c}$',
    }
    model_names = {
        'medium-345M': '$\mathtt{medium}$',
        'xl-1542M': '$\mathtt{xl}$',
        'small-117M': '$\mathtt{small}$',
        'large-762M': '$\mathtt{large}$',
    }
    model_order = {
        'small-117M': 0,
        'medium-345M': 1,
        'large-762M': 2,
        'xl-1542M': 3,
    }
    distribution_order = {
        'text_corr': 1,
        'cluster_corr': 0,
    }

    df = pd.melt(df, id_vars=['model', 'representation', 'representation_type', 'seed'], var_name='distribution', value_name='Correlation',
                 value_vars=['text_corr', 'cluster_corr'])
    df['Distribution'] = df['distribution'].apply(lambda x: legend_names[x])
    df['Correlation (%)'] = df['Correlation'] * 100
    df['Representation'] = df['representation']
    df['Model'] = df['model'].apply(lambda x: model_names[x])
    df['model_order'] = df['model'].apply(lambda x: model_order[x])
    df['distribution_order'] = df['distribution'].apply(lambda x: distribution_order[x])
    df.sort_values(['model_order', 'distribution_order'], inplace=True)

    # fig = plt.figure()
    fig = plt.figure(figsize=(6.4, 3.7))
    ax = sns.barplot(x="Model", y="Correlation (%)", hue="Distribution", data=df)


    plt.ylim([-50, 95])
    plt.legend(loc='lower right', ncol=3, handletextpad=.5, columnspacing=1.2)

    fname = os.path.join(results_fname)
    fig.savefig(fname, bbox_inches='tight')


def load_and_plot_correlations(correlations_fpath, results_fname):
    df = pd.read_csv(correlations_fpath, sep='\t', index_col=0)
    df = df[df.representation == 'gpt2-xl']

    plot_correlations(df, results_fname)


def main():
    args = get_args()

    load_and_plot_correlations(args.correlations_fpath, args.results_fname)


if __name__ == '__main__':
    main()

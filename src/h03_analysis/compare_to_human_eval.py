import os
import sys
import argparse
import pandas as pd

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from util import constants


MODEL_NAMES = {
    "('gpt2', 'p0.9')": 'small-117M-p0.9',
    "('gpt2', 'p1.0')": 'small-117M',
    "('gpt2-large', 'p0.95')": 'large-762M-p0.95',
    "('gpt2-large', 'p1.0')": 'large-762M',
    "('gpt2-medium', 'p0.9')": 'medium-345M-p0.9',
    "('gpt2-medium', 'p1.0')": 'medium-345M',
    "('gpt2-xl', 'p0.95')": 'xl-1542M-p0.95',
    "('gpt2-xl', 'p1.0')": 'xl-1542M',
}


def get_args():
    parser = argparse.ArgumentParser()
    # Save
    parser.add_argument("--results-fpath", type=str, required=True)
    parser.add_argument("--correlations-fpath", type=str, required=True)

    args = parser.parse_args()
    print(args)
    return args


def get_original_bradley_terry_weights(ans_col):
    questions = {
        'Answer.q1': 'interesting',
        'Answer.q2': 'sensible',
        'Answer.q3': 'human-like',
    }

    return constants.BT_SCORES[questions[ans_col]]


def add_bradley_terry_scores_to_dataframe(df_mauve):
    questions = {
        'interesting': 'Answer.q1',
        'sensible': 'Answer.q2',
        'human-like': 'Answer.q3',
    }

    for question, ans_col in questions.items():
        bt_probs = get_original_bradley_terry_weights(ans_col=ans_col)

        bt_probs = {
            MODEL_NAMES[model]: prob for model, prob in bt_probs.items()
            if model in MODEL_NAMES
        }

        # pylint: disable=cell-var-from-loop
        df_mauve[f'bt_{question}'] = df_mauve['model'].apply(lambda x: bt_probs[x])

    return df_mauve


def compare_scores(results_fpath, correlations_fpath):
    df_mauve = pd.read_csv(results_fpath, sep='\t')
    del df_mauve['Unnamed: 0']

    df_mauve = add_bradley_terry_scores_to_dataframe(df_mauve)
    df_mauve.sort_values(['model', 'estimator', 'seed'], inplace=True)

    human_score_names = ['bt_interesting', 'bt_sensible', 'bt_human-like']
    model_score_names = ['auc_score', 'js_score', 'forward_score',
                         'backward_score', 'exponentiated_score']

    corrs = []
    results = []
    for estimator in df_mauve.estimator.unique():
        for seed in df_mauve.seed.unique():
            df_mauve_rep = \
                df_mauve[(df_mauve.estimator == estimator) & (df_mauve.seed == seed)]
            estimator_type = df_mauve_rep.estimator_type.iloc[0]

            corrs += [df_mauve_rep.corr('spearman')[['bt_interesting', 'bt_sensible',
                                                     'bt_human-like']]]
            human_corrs = df_mauve_rep.corr('spearman')[['bt_interesting', 'bt_sensible',
                                                         'bt_human-like']]

            result = {
                'estimator': estimator,
                'estimator_type': estimator_type,
                'seed': seed,
            }
            result.update({
                f'{model_score}-{human_score}': abs(human_corrs[human_score][model_score])
                for human_score in human_score_names
                for model_score in model_score_names
            })
            results += [result]

    df = pd.DataFrame.from_dict(results)
    df.to_csv(correlations_fpath, sep='\t')


def main():
    args = get_args()

    compare_scores(args.results_fpath, args.correlations_fpath)


if __name__ == '__main__':
    main()

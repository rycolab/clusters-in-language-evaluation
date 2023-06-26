import os
import sys
import argparse
import math
import numpy as np
from scipy import stats

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from util import util


def get_args():
    parser = argparse.ArgumentParser()
    # Data type
    parser.add_argument("--data-type", type=str, required=True)
    # Results
    parser.add_argument("--surprisals-model-fpath", type=str, required=True)
    parser.add_argument("--surprisals-human-fpath", type=str, required=True)
    # Save
    parser.add_argument("--results-fpath", type=str, required=True)

    args = parser.parse_args()
    print(args)
    return args


def load_surprisals(surprisals_fpath):
    data = util.read_data(surprisals_fpath)
    return (np.array(data['model_text']['ids']), data['model_text']['surprisals']), \
        (np.array(data['human_text']['ids']), data['human_text']['surprisals'])


def load_all_surprisals(surprisals_model_fpath, surprisals_human_fpath):
    (ids_model_on_model, surprisals_model_on_model), \
        (ids_model_on_human, surprisals_model_on_human) = \
        load_surprisals(surprisals_model_fpath)
    (ids_human_on_model, surprisals_human_on_model), \
        (ids_human_on_human, surprisals_human_on_human) = \
        load_surprisals(surprisals_human_fpath)

    assert (ids_model_on_model == ids_human_on_model).all()
    assert (ids_model_on_human == ids_human_on_human).all()
    assert ids_model_on_model.shape == ids_human_on_human.shape

    return {
        'p_surps': {
            'p_text': surprisals_model_on_model,
            'q_text': surprisals_model_on_human,
        },
        'q_surps': {
            'p_text': surprisals_human_on_model,
            'q_text': surprisals_human_on_human,
        }
    }


def kl_estimate(p_surps, q_surps, mean_type='sample'):
    assert p_surps.shape == q_surps.shape

    if mean_type == 'sample':
        return np.mean(q_surps - p_surps)
    if mean_type == 'truncated':
        return stats.trim_mean(q_surps - p_surps, proportiontocut=.05)

    raise ValueError(f'Invalid mean_type {mean_type}')


def get_mixture_probabilities(p_surps, q_surps, mixture_weight):
    min_surps = np.minimum(p_surps, q_surps)

    mixture = - (
        np.log(mixture_weight * np.exp(- p_surps + min_surps) +
               (1 - mixture_weight) * np.exp(- q_surps + min_surps)
               ) - min_surps)
    return mixture


def get_mixtures(p_surps, q_surps, mixture_weights):
    mixtures = []
    for weight in np.sort(mixture_weights):
        mixture = get_mixture_probabilities(p_surps, q_surps, weight)
        mixtures += [mixture]

    return mixtures


def get_js_divergence(surprisals):
    mixture_ratio = .5

    mixture_p = get_mixture_probabilities(
        surprisals['p_surps']['p_text'], surprisals['q_surps']['p_text'], mixture_ratio)
    mixture_q = get_mixture_probabilities(
        surprisals['p_surps']['q_text'], surprisals['q_surps']['q_text'], mixture_ratio)

    kl_p = kl_estimate(surprisals['p_surps']['p_text'], mixture_p)
    kl_q = kl_estimate(surprisals['q_surps']['q_text'], mixture_q)

    return 0.5 * (kl_q + kl_p)


def get_exponentiated_divergence(surprisals):
    kl_p = kl_estimate(surprisals['p_surps']['p_text'], surprisals['q_surps']['p_text'])
    return math.exp(-.1 * kl_p)


def get_forward_divergence(surprisals):
    kl_p = kl_estimate(surprisals['p_surps']['p_text'], surprisals['q_surps']['p_text'])
    return kl_p


def get_backward_divergence(surprisals):
    kl_q = kl_estimate(surprisals['q_surps']['q_text'], surprisals['p_surps']['q_text'])
    return kl_q


def save_score(scores, fpath):
    data = {
        'js_score': scores['js'],
        'exponentiated_score': scores['exponentiated'],
        'forward_score': scores['forward'],
        'backward_score': scores['backward'],
        'divergences': '-',
    }
    util.write_data(fpath, data)


def get_scores(surprisals_model_fpath, surprisals_human_fpath):
    surprisals = \
        load_all_surprisals(surprisals_model_fpath, surprisals_human_fpath)

    js_score = get_js_divergence(surprisals)
    exponentiated_score = get_exponentiated_divergence(surprisals)
    forward_score = get_forward_divergence(surprisals)
    backward_score = get_backward_divergence(surprisals)
    print(f'The divergence score is {js_score:.4f}')

    return {
        'js': js_score,
        'exponentiated': exponentiated_score,
        'forward': forward_score,
        'backward': backward_score,
    }


def main():
    args = get_args()

    scores = get_scores(
        args.surprisals_model_fpath, args.surprisals_human_fpath)
    save_score(scores, args.results_fpath)


if __name__ == '__main__':
    main()

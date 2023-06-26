import os
import sys
import argparse
import numpy as np
from scipy import stats
from sklearn.metrics import auc
from mauve.compute_mauve import get_divergence_curve_for_multinomials

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from h02_learn.eval import load_checkpoints
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


def get_mixture_probabilities(p_surps, q_surps, mixture_weights):
    min_surps = np.minimum(p_surps, q_surps)

    mixtures = []
    for weight in np.sort(mixture_weights):
        mixture = - (
            np.log(weight * np.exp(- p_surps + min_surps) +
                   (1 - weight) * np.exp(- q_surps + min_surps)
                   ) - min_surps)
        mixtures += [mixture]

    return mixtures


def get_divergence_curve(surprisals, mixture_weights, scaling_factor):
    divergence_curve = [[0, np.inf]]  # extreme point

    mixtures = {
        'p_text': get_mixture_probabilities(
            surprisals['p_surps']['p_text'], surprisals['q_surps']['p_text'],
            mixture_weights),
        'q_text': get_mixture_probabilities(
            surprisals['p_surps']['q_text'], surprisals['q_surps']['q_text'],
            mixture_weights)
    }

    for i in range(mixture_weights.shape[0]):
        mixture_p = mixtures['p_text'][i]
        mixture_q = mixtures['q_text'][i]

        kl_p = kl_estimate(surprisals['p_surps']['p_text'], mixture_p, mean_type='truncated')
        kl_q = kl_estimate(surprisals['q_surps']['q_text'], mixture_q, mean_type='truncated')
        divergence_curve.append([kl_q, kl_p])

    divergence_curve.append([np.inf, 0])  # other extreme point
    divergence_curve = np.asarray(divergence_curve)

    # assert (divergence_curve >= 0).all()
    return np.exp(-scaling_factor * divergence_curve)


def get_mauve_score(divergences):
    kl_p, kl_q = divergences.T
    idxs1 = np.argsort(kl_p)
    idxs2 = np.argsort(kl_q)
    mauve_score = 0.5 * (
        auc(kl_p[idxs1], kl_q[idxs1]) +
        auc(kl_q[idxs2], kl_p[idxs2])
    )

    return mauve_score


def save_score(score, divergences, fpath):
    data = {
        'score': score,
        'divergences': divergences,
    }
    util.write_data(fpath, data)


def get_score(surprisals_model_fpath, surprisals_human_fpath, data_type):
    surprisals = \
        load_all_surprisals(surprisals_model_fpath, surprisals_human_fpath)

    divergence_curve_discretization_size = 25
    if data_type == 'text':
        scaling_factor = .2
    elif data_type == 'cluster':
        scaling_factor = 5.0

    mixture_weights = np.linspace(1e-6, 1-1e-6, divergence_curve_discretization_size)

    divergences = get_divergence_curve(
        surprisals, mixture_weights, scaling_factor)
    mauve_score = get_mauve_score(divergences)
    print(f'The mauve score is {mauve_score:.2f}')

    return mauve_score, divergences


def main():
    args = get_args()

    score, divergences = get_score(
        args.surprisals_model_fpath, args.surprisals_human_fpath, args.data_type)
    save_score(score, divergences, args.results_fpath)


if __name__ == '__main__':
    main()

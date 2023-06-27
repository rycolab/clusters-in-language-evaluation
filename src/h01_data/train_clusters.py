import os
import sys
import argparse
import faiss
from mauve.compute_mauve import get_kmeans_clusters_from_feats

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from util import util


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p-representations-fpath", help="The path to read p-representations data",
                        type=str, required=True)
    parser.add_argument("--q-representations-fpath", help="The path to read q-representations data",
                        type=str, required=True)
    parser.add_argument("--kmeans-fpath", help="The path to save the kmeans model",
                        type=str, required=True)
    parser.add_argument("--pca-fpath", help="The path to save the pca model",
                        type=str, required=True)
    parser.add_argument("--num-buckets", type=int, default=500)
    parser.add_argument("--kmeans-explained-var", type=float, default=.9)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(args)

    return args


def read_representations(fpath):
    data = util.read_data(fpath)
    return data['ids'], data['representations']


def train_clusters(p_representations, q_representations, num_buckets, kmeans_explained_var, seed, norm=None):
    _, _, clustering_model = get_kmeans_clusters_from_feats(
        p_representations, q_representations, num_clusters=num_buckets,
        explained_variance=kmeans_explained_var, norm=norm, whiten=False,
        num_redo=5, max_iter=500, seed=seed)

    return clustering_model


def save_model(model, kmeans_fpath, pca_fpath):
    faiss.write_index(model['kmeans'].index, kmeans_fpath)
    del model['kmeans']
    util.write_data(pca_fpath, model)


def process_data(p_representations_fpath, q_representations_fpath, kmeans_fpath,
                 pca_fpath, num_buckets, kmeans_explained_var, seed):
    # pylint: disable=too-many-arguments,too-many-locals
    _, p_representations = read_representations(p_representations_fpath)
    _, q_representations = read_representations(q_representations_fpath)
    assert (p_representations != q_representations).any(), \
        'Representations shouldnt be the same when getting clusters'

    clustering_model = train_clusters(
        p_representations, q_representations, num_buckets, kmeans_explained_var, seed)

    save_model(clustering_model, kmeans_fpath, pca_fpath)


def main():
    args = get_args()
    process_data(args.p_representations_fpath, args.q_representations_fpath, args.kmeans_fpath,
                 args.pca_fpath, args.num_buckets, args.kmeans_explained_var, args.seed)


if __name__ == '__main__':
    main()

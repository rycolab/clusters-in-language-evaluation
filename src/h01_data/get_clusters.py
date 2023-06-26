import os
import sys
import argparse
import numpy as np
import faiss

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from h01_data.train_clusters import read_representations
from util import util


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--representations-fpath", help="The path to read representations data",
                        type=str, required=True)
    parser.add_argument("--clusters-fpath", help="The path to save clusters data",
                        type=str, required=True)
    parser.add_argument("--kmeans-fpath", help="The path to save the kmeans model",
                        type=str, required=True)
    parser.add_argument("--pca-fpath", help="The path to save the pca model",
                        type=str, required=True)
    parser.add_argument("--num-buckets", type=int, default=500)
    parser.add_argument("--kmeans-explained-var", type=float, default=.9)
    args = parser.parse_args()
    print(args)

    return args


def load_model(kmeans_fpath, pca_fpath):
    model = util.read_data(pca_fpath)
    kmeans_index = faiss.read_index(kmeans_fpath)
    model['kmeans'] = kmeans_index

    return model


def get_clusters(pca, kmeans, dimensionality, representations):
    reps_pca = pca.transform(representations)[:, :dimensionality]
    reps_pca = reps_pca.astype(np.float32)

    _, labels = kmeans.search(reps_pca, 1)
    labels = labels.reshape(-1)

    return reps_pca, labels


def save_clusters(ids, clusters, fpath):
    data = {
        'ids': ids,
        'clusters': clusters,
    }
    util.write_data(fpath, data)


def process_data(representations_fpath, clusters_fpath, kmeans_fpath,
                 pca_fpath):
    ids, representations = read_representations(representations_fpath)
    clustering_model = load_model(kmeans_fpath, pca_fpath)

    _, clusters = get_clusters(
        clustering_model['pca'], clustering_model['kmeans'],
        clustering_model['dimensionality'], representations)

    save_clusters(ids, clusters, clusters_fpath)


def main():
    args = get_args()
    process_data(args.representations_fpath, args.clusters_fpath,
                 args.kmeans_fpath, args.pca_fpath)


if __name__ == '__main__':
    main()

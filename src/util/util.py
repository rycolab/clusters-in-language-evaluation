import os
import io
import csv
import shutil
import pathlib
import pickle
import random
from tqdm import tqdm
import numpy as np
import torch


def config(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def write_csv(filename, results):
    with io.open(filename, 'w', encoding='utf8') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(results)


def write_data(filename, data):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def read_data(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data


def rmdir_if_exists(fdir):
    if os.path.exists(fdir):
        shutil.rmtree(fdir)


def file_len(fname):
    if not os.path.isfile(fname):
        return 0
    return sum(1 for _ in open(fname, 'r', encoding='utf8'))


def mkdir(folder):
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def try_to_load_results(fpath, load_function):
    if os.path.isfile(fpath):
        return load_function(fpath)
    return None, None


def process_and_save_temp(ids, texts, function, dump_size, function_kwargs,
                          load_function, save_function, temp_fpath):
    _, results = try_to_load_results(temp_fpath, load_function)

    done_size = len(results) if results is not None else 0
    with tqdm(total=len(texts), desc='Getting results',
              initial=done_size, dynamic_ncols=True) as pbar:
        for texts_chunk in chunker(texts[done_size:], dump_size):
            results_chunk = function(
                texts=texts_chunk, **function_kwargs)

            results = \
                results_chunk if results is None else \
                np.concatenate([results, results_chunk], axis=0)

            save_function(ids, results, temp_fpath)
            pbar.update(len(texts_chunk))

    return results

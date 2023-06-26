import os
import sys
import argparse
import torch
from sklearn.preprocessing import normalize
from mauve.compute_mauve import get_features_from_input

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from util import constants
from util import util


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-fpath", help="The path to save text data",
                        type=str, required=True)
    parser.add_argument("--representations-fpath", help="The path to save text data",
                        type=str, required=True)
    parser.add_argument("--representations-model", help="The model to use to get representations",
                        type=str, required=True)
    parser.add_argument("--representations-type",
                        help="The type of representations. Choices: ['final', 'average']",
                        type=str, choices=['final', 'average'], required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dump-size", type=int, default=10000)
    args = parser.parse_args()
    print(args)

    return args


def save_representations(ids, representations, fpath):
    data = {
        'ids': ids,
        'representations': representations,
    }
    util.write_data(fpath, data)


def load_representations(fpath):
    data = util.read_data(fpath)
    return data['ids'], data['representations']


def try_load_representations(fpath):
    if os.path.isfile(fpath):
        return load_representations(fpath)
    return None, None


def process_data(text_fpath, representations_fpath, representations_model,
                 representations_type, batch_size, dump_size, norm='l2'):
    text_data = util.read_data(text_fpath)
    ids = text_data['ids']
    texts = text_data['texts']
    is_mean = representations_type == 'average'

    max_len = 512 if 'bert' in representations_model else 1024

    # If dump_size is -1 save all results only once, else save temporary results
    if dump_size != -1:
        representations_fpath_temp = representations_fpath + '.temp'
        featurize_kwargs = {
            'featurize_model_name': representations_model, 'is_mean': is_mean, 'max_len': max_len,
            'batch_size': batch_size, 'name': 'p', 'device_id': 0, 'verbose': False,
            'features': None, 'tokenized_texts': None,
        }
        representations = util.process_and_save_temp(
            ids, texts, function=get_features_from_input, dump_size=dump_size,
            function_kwargs=featurize_kwargs, load_function=load_representations,
            save_function=save_representations, temp_fpath=representations_fpath_temp)
    else:
        representations = get_features_from_input(
            texts=texts, featurize_model_name=representations_model, is_mean=is_mean,
            max_len=1024, batch_size=batch_size, name='p', device_id=0, verbose=False,
            features=None, tokenized_texts=None)

    if norm in ['l2', 'l1']:
        representations = normalize(representations, norm=norm, axis=1)
    save_representations(ids, representations, representations_fpath)


def main():
    args = get_args()
    print('Is PyTorch in cuda? ', torch.cuda.is_available())
    print('Is PyTorch device: ', constants.device)

    process_data(args.text_fpath, args.representations_fpath,
                 args.representations_model, args.representations_type,
                 args.batch_size, args.dump_size)


if __name__ == '__main__':
    main()

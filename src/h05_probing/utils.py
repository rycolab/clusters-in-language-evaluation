import json
from mauve.compute_mauve import get_features_from_input
from nltk.tokenize import treebank

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from h01_data.get_representations import save_representations, load_representations


def clean(text):
    cleaned = []
    treebank_detokenizer = treebank.TreebankWordDetokenizer()
    for t in text:
        t = treebank_detokenizer.detokenize(t.replace('<|endoftext|>', '').strip().split())
        if len(t.split()) < 10:
            continue
        cleaned.append(t)
    return cleaned

def load_gpt2_dataset(json_file_name, num_examples=float('inf')):
    texts = []
    for i, line in enumerate(open(json_file_name)):
        if i >= num_examples:
            break
        texts.append(json.loads(line)['text'])
    return texts

def get_representations(texts, model, mean, recompute=False, save=True, save_dir="reps", max_len=512):
    def get_repfile_name(t, model, mean):
        mean_flag = '_m' if mean else ''
        return os.path.join(save_dir, model.split("/")[-1] + mean_flag + str(hash(t)))
    file_name = get_repfile_name(texts, model, mean)
    features = None
    if not recompute:
        try:
            _, features = load_representations(file_name)
        except FileNotFoundError:
            pass
    if features is None:           
        features = get_features_from_input(
                    texts=texts, featurize_model_name=model, is_mean=mean,
                    max_len=max_len, batch_size=4, name='p', device_id=0, verbose=False,
                    features=None, tokenized_texts=None)
    if save:
        save_representations(list(range(len(texts))), features, file_name)
    return features

import os
import sys
import argparse
import json
import pandas as pd

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from util import util


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-fpath", help="The path of the raw data",
                        type=str, required=True)
    parser.add_argument("--text-fpath", help="The path to save text data",
                        type=str, required=True)
    parser.add_argument("--baretext-fpath", help="The path to save bare text data",
                        type=str, required=True)
    parser.add_argument("--base-info-fpath", help="The path to save base info data",
                        type=str, required=True)
    parser.add_argument("--max-sentences", type=int, default=25000)
    args = parser.parse_args()
    print(args)

    return args


def read_data(src_fpath, max_sentences):
    ids, texts, endeds, lengths = [], [], [], []
    with open(src_fpath, 'rb') as f:
        for line in f:
            data = json.loads(line)

            ids += [data['id']]
            texts += [data['text']]
            endeds += [data['ended']]
            lengths += [data['length']]

            if len(texts) >= max_sentences:
                break

    return ids, texts, endeds, lengths


def save_base_info(ids, endeds, lengths, fpath):
    df = pd.DataFrame({
        'id': ids,
        'ended': endeds,
        'length': lengths,
    })
    df.to_csv(fpath, sep='\t')


def save_text(ids, texts, fpath):
    text_data = {
        'ids': ids,
        'texts': texts,
    }
    util.write_data(fpath, text_data)


def save_baretext(texts, fpath):
    with open(fpath, 'w', encoding='utf8') as f:
        for sentence in texts:
            # Remove inner line breaks
            f.write("%s\n" %
                    sentence.replace('\n\n', ' ').replace('\n', ' '))


def process_data(src_fpath, text_fpath, base_info_fpath, baretext_fpath, max_sentences):
    ids, texts, endeds, lengths = read_data(src_fpath, max_sentences)

    save_baretext(texts, baretext_fpath)
    save_base_info(ids, endeds, lengths, base_info_fpath)
    save_text(ids, texts, text_fpath)


def main():
    args = get_args()

    process_data(args.raw_data_fpath, args.text_fpath,
                 args.base_info_fpath, args.baretext_fpath,
                 args.max_sentences)


if __name__ == '__main__':
    main()

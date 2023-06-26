"""
Get the surprisals of each sentence based on a specific language model.
This script was partly adapted from code in the repo:
    https://github.com/krishnap25/mauve
"""

import sys
import os
import argparse
from tqdm import tqdm
import torch
from torch.functional import F
from transformers import GPT2LMHeadModel, AutoTokenizer

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from util import constants
from util import util


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-fpath", help="The path to save text data",
                        type=str, required=True)
    parser.add_argument("--surprisals-fpath", help="The path to save text data",
                        type=str, required=True)
    parser.add_argument("--model", help="The model to use to get surprisals",
                        type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dump-size", type=int, default=10000)
    args = parser.parse_args()
    print(args)

    return args


def get_model(model_name, tokenizer):
    if 'gpt2' in model_name:
        model = GPT2LMHeadModel.from_pretrained(
            model_name, pad_token_id=tokenizer.eos_token_id)
        model = model.to(constants.device)
        model = model.eval()
    else:
        raise ValueError(f'Unknown model: {model_name}')
    return model


def get_tokenizer(model_name='gpt2'):
    if 'gpt2' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        raise ValueError(f'Unknown model: {model_name}')
    return tokenizer


def count_parameters(model):
    return sum(x.numel() for x in model.parameters() if x.requires_grad)


def get_batch_surprisals(model, batch):
    padded_batch = torch.nn.utils.rnn.pad_sequence(
        batch, batch_first=True, padding_value=0).to(constants.device)
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [torch.ones(sent.size(0)) for sent in batch],
        batch_first=True, padding_value=0).long().to(constants.device)

    outs = model(input_ids=padded_batch, attention_mask=attention_mask,
                 past_key_values=None, return_dict=True)

    # Get surprisals of following word pieces
    surprisals_full = - F.log_softmax(outs.logits, dim=-1)
    surprisals_sentence = torch.gather(
        surprisals_full[:, :-1], -1, padded_batch[:, 1:].unsqueeze(-1)).squeeze(-1)
    surprisals_sentence = (surprisals_sentence * attention_mask[:, 1:])

    return surprisals_sentence.sum(-1).cpu()


@torch.no_grad()
def get_surprisals_from_tokenised_text(model, tokenized_texts, batch_size):
    surprisals = []

    with tqdm(total=len(tokenized_texts), desc="Getting chunk surprisals") as pbar:
        for batch in util.chunker(tokenized_texts, batch_size):
            surprisals_batch = get_batch_surprisals(model, batch)

            surprisals.append(surprisals_batch)
            pbar.update(len(batch))

    return torch.cat(surprisals).numpy()


def get_surprisals_from_input(texts, model_name, batch_size, max_len=1024):
    tokenizer = get_tokenizer(model_name)
    tokenized_texts = [
        tokenizer.encode(
            tokenizer.bos_token + sen + tokenizer.eos_token,
            return_tensors='pt', truncation=True, max_length=max_len).view(-1)
        for sen in texts
    ]

    model = get_model(model_name, tokenizer)
    # print('Model %s has %dM parameters' % (model_name, count_parameters(model) * 1e-6))

    surprisals = get_surprisals_from_tokenised_text(
        model, tokenized_texts, batch_size=batch_size)

    return surprisals


def save_surprisals(ids, surprisals, fpath):
    data = {
        'ids': ids,
        'surprisals': surprisals,
    }
    util.write_data(fpath, data)


def load_surprisals(fpath):
    data = util.read_data(fpath)
    return data['ids'], data['surprisals']


def try_load_surprisals(fpath):
    if os.path.isfile(fpath):
        return load_surprisals(fpath)
    return None, None


def process_data(text_fpath, surprisals_fpath, model, batch_size, dump_size):
    text_data = util.read_data(text_fpath)
    ids = text_data['ids']
    texts = text_data['texts']
    model = constants.MODEL_NAMES[model]

    # If dump_size is -1 save all results only once, else save temporary results
    if dump_size != -1:
        surprisals_fpath_temp = surprisals_fpath + '.temp'
        surprisals = util.process_and_save_temp(
            ids, texts, function=get_surprisals_from_input, dump_size=dump_size,
            function_kwargs={'model_name': model, 'batch_size': batch_size, 'max_len': 1024},
            load_function=load_surprisals, save_function=save_surprisals,
            temp_fpath=surprisals_fpath_temp)
    else:
        surprisals = get_surprisals_from_input(
            texts=texts, model_name=model, batch_size=batch_size, max_len=1024)

    save_surprisals(ids, surprisals, surprisals_fpath)


def main():
    args = get_args()
    print('Is PyTorch in cuda? ', torch.cuda.is_available())
    print('Is PyTorch device: ', constants.device)

    process_data(args.text_fpath, args.surprisals_fpath,
                 args.model, batch_size=args.batch_size, dump_size=args.dump_size)


if __name__ == '__main__':
    main()

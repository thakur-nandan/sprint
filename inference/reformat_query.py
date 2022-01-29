import argparse
import json
import os
from posixpath import split
import tqdm
from .evaluate import load_qrels

def beir(data_dir, output_dir=None):
    if output_dir is None:
        output_dir = data_dir
    else:
        os.makedirs(output_dir, exist_ok=True)

    splits = os.listdir(os.path.join(data_dir, 'qrels'))
    splits = list(map(lambda fname: fname.replace('.tsv', ''), splits))
    qrels_splits = {split: load_qrels(os.path.join(data_dir, 'qrels', f'{split}.tsv')) for split in splits}
    qid_to_split = {qid: split for split, qrels in qrels_splits.items() for qid in qrels}
    fouts = {split: open(os.path.join(output_dir, f'queries-{split}.reformatted.tsv'), 'w') for split in splits}
    with open(os.path.join(data_dir, 'queries.jsonl'), 'r') as fin:
        for line in tqdm.tqdm(fin):
            line_dict = json.loads(line)
            qid = line_dict['_id']
            line_output = '\t'.join([line_dict['_id'], line_dict['text']]) + '\n'
            fouts[qid_to_split[qid]].write(line_output)


def run(original_format, data_dir, output_dir=None):
    original_format = original_format.lower()

    if original_format == 'beir':
        beir(data_dir, output_dir)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_format')
    parser.add_argument('--data_dir')
    parser.add_argument('--output_dir', default=None)
    args = parser.parse_args()
    run(**vars(args))

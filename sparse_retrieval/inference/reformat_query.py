import argparse
from email import header
import json
import os
from typing import Dict
from collections import defaultdict
import shutil
import tqdm
from .evaluate import load_qrels

def beir(data_dir, output_dir=None):
    if output_dir is None:
        output_dir = data_dir
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    qid_mapping: Dict[str, str] = defaultdict(str)  # Since pyserini does not support string query IDs, we need to change them into integers

    splits = os.listdir(os.path.join(data_dir, 'qrels'))
    splits = list(map(lambda fname: fname.replace('.tsv', ''), splits))
    qrels_splits: Dict[str, Dict[str, Dict[str, float]]] = {split: load_qrels(os.path.join(data_dir, 'qrels', f'{split}.tsv')) for split in splits}
    
    need_mapping_to_int = False
    first_split = list(qrels_splits)[0]
    first_qid: str = list(qrels_splits[first_split].keys())[0]
    if first_qid.isnumeric():
        try:
            eval(first_qid)
        except SyntaxError:  # e.g. '0111'
            need_mapping_to_int = True
    else:
        need_mapping_to_int = True

    # TODO: change qids into digits-only style
    qid_to_split = {qid: split for split, qrels in qrels_splits.items() for qid in qrels}
    fouts = {split: open(os.path.join(output_dir, f'queries-{split}.reformatted.tsv'), 'w') for split in splits}
    with open(os.path.join(data_dir, 'queries.jsonl'), 'r') as fin:
        for line in tqdm.tqdm(fin):
            line_dict = json.loads(line)
            qid = line_dict['_id']
            target_split = qid_to_split[qid]

            if need_mapping_to_int:
                if qid in qid_mapping:
                    qid_new = qid_mapping[qid]
                else:
                    qid_new = str(len(qid_mapping))
                    qid_mapping[qid] = qid_new
                qid = qid_new

            line_output = '\t'.join([qid, line_dict['text']]) + '\n'
            fouts[target_split].write(line_output)

    if need_mapping_to_int:
        for split in splits:
            path_qrels_org = os.path.join(data_dir, 'qrels', f'{split}.tsv')
            path_qrels_bak = os.path.join(data_dir, 'qrels', f'{split}.bak.tsv')
            shutil.copyfile(path_qrels_org, path_qrels_bak)

            qrels = qrels_splits[split]
            with open(path_qrels_org, 'w') as f:
                header = '\t'.join(['query-id', 'corpus-id', 'score']) + '\n' 
                f.write(header)
                for qid, rels in qrels.items():
                    assert qid in qid_mapping, "error: it seems that qrels and queries files do not match"
                    qid_new = qid_mapping[qid]
                    for doc_id, score in rels.items():
                        line = '\t'.join([qid_new, doc_id, str(score)]) + '\n'
                        f.write(line)


def run(original_format, data_dir, output_dir=None):
    original_format = original_format.lower()

    if original_format == 'beir':
        beir(data_dir, output_dir)
    else:
        raise NotImplementedError
    
    print('Done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_format')
    parser.add_argument('--data_dir')
    parser.add_argument('--output_dir', default=None)
    args = parser.parse_args()
    run(**vars(args))

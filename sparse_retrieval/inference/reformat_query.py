import argparse
import os
from .utils import load_qrels, load_queries

def convert_beir_queries(data_dir, topic_split, output_dir=None):
    if output_dir is None:
        output_dir = data_dir
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    queries = load_queries(os.path.join(data_dir, "queries.jsonl"))
    qrels = load_qrels(os.path.join(data_dir, 'qrels', f'{topic_split}.tsv'))
    queries = {qid: queries[qid] for qid in qrels}
    
    with open(os.path.join(output_dir, f'queries-{topic_split}.tsv'), 'w') as fOut:
        for query_id, query in queries.items():
            line = '\t'.join([str(query_id), query]) + '\n'
            fOut.write(line)

def run(original_format, data_dir, topic_split, output_dir=None):
    original_format = original_format.lower()

    if original_format == 'beir':
        convert_beir_queries(data_dir, topic_split, output_dir)
    else:
        raise NotImplementedError
    
    print(f'{__name__}: Done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_format')
    parser.add_argument('--data_dir')
    parser.add_argument('--topic_split')
    parser.add_argument('--output_dir', default=None)
    args = parser.parse_args()
    run(**vars(args))

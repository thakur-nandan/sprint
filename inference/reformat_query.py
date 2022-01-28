import argparse
import json
import os
import tqdm

def beir(data_dir, output_dir=None):
    if output_dir is None:
        output_dir = data_dir
    else:
        os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, 'queries.reformatted.tsv')
    with open(os.path.join(data_dir, 'queries.jsonl'), 'r') as fin, open(output_path, 'w') as fout:
        for line in tqdm.tqdm(fin):
            line_dict = json.loads(line)
            line_output = '\t'.join([line_dict['_id'], line_dict['text']]) + '\n'
            fout.write(line_output)

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

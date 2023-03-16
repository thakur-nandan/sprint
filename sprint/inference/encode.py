from multiprocessing import Value
from typing import Callable, List, Iterable, Tuple, Union
from torch import multiprocessing as mp
try:
     mp.set_start_method('spawn')  # needed by CUDA + multiprocessing
except RuntimeError:
    pass
import torch
import gzip
import os
import math
import json
import tqdm
import argparse
import logging
import copy
from . import data_iters
from . import encoder_builders
from pyserini.encode import QueryEncoder, DocumentEncoder

logger = logging.getLogger(__name__)

def one_process(
    encoder_builder: Callable[[int], Union[QueryEncoder, DocumentEncoder]], 
    data_iter: Iterable,
    gpu: int,
    output_dir: str,
    data_range: Tuple[int, int],
    batch_size: int,
    chunk_size: int,
    return_value: Value,
    show_progress_bar: bool
):
    nlines_encoded = 0
    encoder = encoder_builder(gpu)
    batch_ids = []
    batch_texts = []
    term_weights_chunk = []
    begin, end = data_range
    for i, example in enumerate(tqdm.tqdm(data_iter, disable=not show_progress_bar, total=end-begin)):
        if not (begin <= i < end):
            continue
        
        assert type(example) == dict, f'Example must be a dict, but got {type(example)}'
        assert 'id' in example, f'Example must have an id, but got {example}'
        assert 'text' in example, f'Example must have a text, but got {example}'

        batch_ids.append(example['id'])
        batch_texts.append(example['text'])

        if len(batch_ids) >= batch_size or len(batch_ids) + len(term_weights_chunk) >= chunk_size or i == end - 1:
            with torch.no_grad():
                term_weights_batch = encoder.encode(batch_texts)

            term_weights_batch = [
                {
                    "id": doc_id, 
                    "contents": text, 
                    "vector": term_weights
                } for doc_id, text, term_weights in zip(batch_ids, batch_texts, term_weights_batch)
            ]
            term_weights_chunk.extend(term_weights_batch)

            if len(term_weights_chunk) >= chunk_size or i == end - 1:
                chunk_idx = math.ceil((i + 1) / chunk_size) - 1
                fname_save = f'split{chunk_idx}.jsonl.gz'
                logger.info(f'Writing to {fname_save}. i: {i}, range: {data_range}')
                
                os.makedirs(output_dir, exist_ok=True)
                with gzip.open(os.path.join(output_dir, fname_save), 'at+') as f:
                    for line in term_weights_chunk:
                        f.write(json.dumps(line) + '\n')
                        nlines_encoded += 1
                
                term_weights_chunk = []

            batch_ids = []
            batch_texts = []
    
    logger.info(f'Encoded {nlines_encoded} lines. The range is {data_range}')
    return_value.value = nlines_encoded

def _run(
    encoder_builder: Callable[[int], Union[QueryEncoder, DocumentEncoder]],
    data_iter: Iterable, 
    gpus: List[int], 
    output_dir: str,
    batch_size: int, 
    chunk_size: int
):
    nexamples = len(data_iter)
    assert len(set(gpus)) == len(gpus)
    if os.path.exists(output_dir):
        assert len(os.listdir(output_dir)) == 0, 'Please empty the output_dir'

    range_size = nexamples // len(gpus)
    chunk_size = min(chunk_size, range_size-1)  # important for saving all the encoded data !!
    
    ranges = [(b, b+range_size) for b in range(0, nexamples, range_size)]
    ranges[len(gpus)-1] = (ranges[len(gpus)-1][0], ranges[-1][1])  # merge all the leftovers into the last range
    ranges = ranges[:len(gpus)]  # keep only #GPUs ranges

    ranges[-1] = (ranges[-1][0], min(ranges[-1][1], nexamples))  # make sure the last range is right, which is important for the criterion of ending point
    return_values = [Value('d') for _ in range(len(ranges))]
    if len(gpus) > 1:
        ps = [mp.Process(
                target=one_process, 
                args=(
                    encoder_builder, 
                    copy.deepcopy(data_iter),
                    gpu,
                    output_dir,
                    data_range,
                    batch_size,
                    chunk_size,
                    return_values[i],
                    gpu == gpus[0]
                )
            ) for i, (gpu, data_range) in enumerate(zip(gpus, ranges))]
        
        for p in ps:
            p.start()
        for p in ps:
            p.join()
    else:
        # especially for debugging usage
        one_process(
            encoder_builder, 
            data_iter,
            gpus[0],
            output_dir,
            (0, len(data_iter)),
            batch_size,
            chunk_size,
            return_values[0],
            True
        )
    
    nlines_encoded = sum([v.value for v in return_values])
    assert nlines_encoded == nexamples, f'Error: Did not encode all the documents ({nlines_encoded}/{nexamples})!!!'

def run(encoder_name, ckpt_name, data_name, data_dir, gpus, output_dir, batch_size=64, chunk_size=100000):
    encoder_builder = encoder_builders.get_builder(encoder_name, ckpt_name, 'document')
    data_iter = data_iters.build(data_name, data_dir)
    torch.multiprocessing.freeze_support()
    _run(encoder_builder, data_iter, gpus, output_dir, batch_size, chunk_size)
    logger.info(f'{__name__}: Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_name', type=str, choices=["unicoil", "splade", "sparta", "tildev2", "deepimpact"], help="The name of the sparse encoder. For example, splade, unicoil, etc.")
    parser.add_argument('--ckpt_name', type=str, help="The name of the sparse model checkpoint.")
    parser.add_argument('--data_name', type=str, help="The name of the dataset. For example, scifact, trec-covid, etc.")
    parser.add_argument('--data_dir', type=str, help="The directory where the BEIR data is present. For example, if you are using the scifact dataset, this is the directory where the corpus.jsonl is stored.")
    parser.add_argument('--gpus', nargs='+', type=int, help="The gpus to use. For example, --gpus 0 1 2 3")
    parser.add_argument('--output_dir', type=str, help="The output directory where the encoded data will be saved.")
    parser.add_argument('--batch_size', type=int, default=64, help="Default is 64. This is the number of documents to be encoded at a time. If you have a large GPU, you can increase this number to speed up the encoding process.")
    parser.add_argument('--chunk_size', type=int, default=100000, help="Default is 100k. This is the number of documents to be encoded at a time. If you have a large GPU, you can increase this number to speed up the encoding process.")
    args = parser.parse_args()
    run(**vars(args))

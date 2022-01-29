from typing import Callable, List, Dict, Iterable, Tuple
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
import copy
import crash_ipdb
from . import data_iters
from . import encode_fn_builders


def one_process(
    encode_fn_builder: Callable[[int], Callable[[List[str],], Dict[str, object]]], 
    data_iter: Iterable,
    gpu: int,
    output_dir: str,
    data_range: Tuple[int, int],
    batch_size: int,
    chunk_size: int,
    show_progress_bar: bool
):
    encode_fn = encode_fn_builder(gpu)
    batch_ids = []
    batch_texts = []
    term_weights_chunk = []
    begin, end = data_range
    for i, example in enumerate(tqdm.tqdm(data_iter, disable=not show_progress_bar, total=end-begin)):
        if not (begin <= i < end):
            continue
        
        assert type(example) == dict
        assert 'id' in example
        assert 'text' in example

        batch_ids.append(example['id'])
        batch_texts.append(example['text'])

        if len(batch_ids) >= batch_size or i == end - 1:
            with torch.no_grad():
                term_weights_batch = encode_fn(batch_texts)

            term_weights_batch = [
                {
                    "id": doc_id, 
                    "contents": text, 
                    "vector": term_weights
                } for doc_id, text, term_weights in zip(batch_ids, batch_texts, term_weights_batch)
            ]
            term_weights_chunk.extend(term_weights_batch)
            
            if len(term_weights_chunk) >= chunk_size or i == end - 1:
                chunk_idx = (i + 1) // chunk_size - 1
                fname_save = f'split{chunk_idx}.jsonl.gz'
                
                os.makedirs(output_dir, exist_ok=True)
                with gzip.open(os.path.join(output_dir, fname_save), 'wt') as f:
                    for line in term_weights_chunk:
                        f.write(json.dumps(line) + '\n')

                term_weights_chunk = []

            batch_ids = []
            batch_texts = []

def _run(
    encode_fn_builder: Callable[[int], Callable[[List[str],], Dict[str, object]]],
    data_iter: Iterable, 
    gpus: List[int], 
    output_dir: str,
    batch_size: int, 
    chunk_size: int
):
    nexamples = len(data_iter)
    assert len(set(gpus)) == len(gpus)
    range_size = math.ceil(nexamples / len(gpus))
    chunk_size = min(chunk_size, range_size-1)  # important for saving all the encoded data !!
    
    ranges = [(b, b+range_size) for b in range(0, nexamples, range_size)]
    ranges[-1] = (ranges[-1][0], min(ranges[-1][1], nexamples))  # make sure the last range is right, which is important for the criterion of ending point
    if len(gpus) > 1:
        ps = [mp.Process(target=one_process, args=(
            encode_fn_builder, 
            copy.deepcopy(data_iter),
            gpu,
            output_dir,
            data_range,
            batch_size,
            chunk_size,
            gpu == gpus[0]
        )) for gpu, data_range in zip(gpus, ranges)]
        
        for p in ps:
            p.start()
        for p in ps:
            p.join()
    else:
        # especially for debugging usage
        one_process(
            encode_fn_builder, 
            data_iter,
            gpus[0],
            output_dir,
            (0, len(data_iter)),
            batch_size,
            chunk_size,
            True
        )

def run(encoder_name, ckpt_name, data_name, data_dir, gpus, output_dir, batch_size=64, chunk_size=100000):
    encode_fn_builder = encode_fn_builders.build(encoder_name, ckpt_name)
    data_iter = data_iters.build(data_name, data_dir)
    _run(encode_fn_builder, data_iter, gpus, output_dir, batch_size, chunk_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_name')
    parser.add_argument('--ckpt_name')
    parser.add_argument('--data_name')
    parser.add_argument('--data_dir')
    parser.add_argument('--gpus', nargs='+', type=int)
    parser.add_argument('--output_dir')
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--chunk_size', default=100000)
    args = parser.parse_args()
    run(**vars(args))

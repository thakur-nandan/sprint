import argparse
from functools import partial
import os
import gzip
import json
from typing import List, Callable
import tqdm
import dill
from multiprocessing import Process, Value
import math
import numpy as np
import crash_ipdb


def one_process(
    fpaths: List[str],
    output_dir: str,
    check_range_only: bool,
    quantize: Callable[[float,], int],
    return_value: Value,
    show_progress_bar: bool
):
    max_value = 0
    for fpath in tqdm.tqdm(fpaths, disable=not show_progress_bar):
        fout = None
        if not check_range_only:
            fpath_output = os.path.join(output_dir, os.path.basename(fpath))
            os.makedirs(output_dir, exist_ok=True)
            fout = gzip.open(fpath_output, 'wt')

        with gzip.open(fpath, 'r') as f:
            for line in f:
                line_dict = json.loads(line)
                term_dict = line_dict['vector']
                max_weight = max(term_dict.values())
                if max_weight > max_value:
                    max_value = max_weight

                if not check_range_only:
                    line_dict['vector'] = dict(map(lambda x: (x[0], quantize(x[1])), term_dict.items()))
                    fout.write(json.dumps(line_dict) + '\n')
        
        if fout is not None:
            fout.close()

    return_value.value = max_value

def range_nbits(original_score_range, quantization_nbits, w):
    return int(np.ceil(w/original_score_range * (2**quantization_nbits)))

def ndigits_round(factor, w):
    return round(w * factor)

def build_quantize_fn(method, original_score_range, quantization_nbits, ndigits):
    if method == 'range-nbits':
        assert type(original_score_range) in [float, int] and original_score_range != 0
        assert type(quantization_nbits) is int and quantization_nbits > 0
        quantize_fn = partial(range_nbits, original_score_range, quantization_nbits)
    elif method == 'ndigits-round':
        assert type(ndigits) is int and ndigits > 0
        factor = 10 ** ndigits
        quantize_fn = partial(ndigits_round, factor)
    else:
        raise NotImplementedError
    
    return quantize_fn

def run(
    collection_dir: str,
    output_dir: str,
    method: str,
    original_score_range: float,
    quantization_nbits: int,
    ndigits: int,
    check_range_only: bool,
    nprocs: int
):
    quantize_fn = None
    if not check_range_only:
        assert type(output_dir) is str
        quantize_fn = build_quantize_fn(method, original_score_range, quantization_nbits, ndigits)
        print(f'Using quantization method: {method}')

    fpaths = [os.path.join(collection_dir, fname) for fname in os.listdir(collection_dir)]
    nprocs = min(len(fpaths), nprocs)
    nfpaths_per_proc = len(fpaths) // nprocs
    ranges = [(b, b+nfpaths_per_proc) for b in range(0, len(fpaths), nfpaths_per_proc)]
    ranges[nprocs-1] = (ranges[nprocs-1][0], ranges[-1][1])  # merge all the leftovers into the last range
    ranges = ranges[:nprocs]  # keep only nprocs ranges
    fpaths_divided = [[fpath for fpath in fpaths[b:e]] for (b, e) in ranges]
    
    return_values = [Value('d') for _ in range(nprocs)]
    processes = [Process(target=one_process, args=(
        fpaths_divided[i],
        output_dir,
        check_range_only,
        quantize_fn,
        return_values[i],
        i == 0
    )) for i in range(nprocs)]    

    [p.start() for p in processes]
    [p.join() for p in processes] 

    max_term_weight = max([v.value for v in return_values])
    print('Max. term weight:', max_term_weight)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--collection_dir')
    parser.add_argument('--output_dir', required=False)
    parser.add_argument('--method', required=False, choices=['range-nbits', 'ndigits-round'])
    # for 'range-nbits':
    parser.add_argument('--original_score_range', type=float, default=5)
    parser.add_argument('--quantization_nbits', type=int, default=8)
    # for 'ndigits-round':
    parser.add_argument('--ndigits', type=int, default=2, help='2 means *100')
    parser.add_argument('--check_range_only', action='store_true')
    parser.add_argument('--nprocs', type=int)
    args = parser.parse_args()
    run(**vars(args))

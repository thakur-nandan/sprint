import argparse
import os
import gzip
import json
from typing import List
import tqdm
from multiprocessing import Process, Manager, Value
import math
import numpy as np


def one_process(
    fpaths: List[str],
    output_dir: str,
    original_score_range: float,
    quantization_nbits: int,
    check_range_only: bool,
    return_value: Value,
    show_progress_bar: bool
):
    max_value = 0
    quantize = lambda w: int(np.ceil(w/original_score_range * (2**quantization_nbits)))
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

def run(
    collection_dir: str,
    output_dir: str,
    original_score_range: float,
    quantization_nbits: int,
    check_range_only: bool,
    nprocs: int
):
    if not check_range_only:
        assert type(output_dir) is str
        assert type(original_score_range) in [float, int]
        assert type(quantization_nbits) is int

    fpaths = [os.path.join(collection_dir, fname) for fname in os.listdir(collection_dir)]
    nprocs = min(len(fpaths), nprocs)
    nfpaths_per_proc = math.ceil(len(fpaths) / nprocs)
    fpaths_divided = [[fpath for fpath in fpaths[b:b+nfpaths_per_proc]] for b in range(0, len(fpaths), nfpaths_per_proc)]
    return_values = [Value('d') for _ in range(nprocs)]
    processes = [Process(target=one_process, args=(
        fpaths_divided[i],
        output_dir,
        original_score_range,
        quantization_nbits,
        check_range_only,
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
    parser.add_argument('--original_score_range', type=float, default=5)
    parser.add_argument('--quantization_nbits', type=int, default=8)
    parser.add_argument('--check_range_only', action='store_true')
    parser.add_argument('--nprocs', type=int)
    args = parser.parse_args()
    run(**vars(args))

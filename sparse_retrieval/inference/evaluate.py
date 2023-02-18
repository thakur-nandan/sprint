import argparse
import csv
import json
import os
from typing import Dict, List

import numpy as np
from .utils import load_qrels, get_processor_name, get_folder_size, bin_and_average, bin_and_std
from beir.retrieval.evaluation import EvaluateRetrieval


def load_results(result_path, format) -> Dict[str, Dict[str, float]]:
    results = {}

    # the format is from pyserini's output-format: 
    # https://github.com/castorini/pyserini/blob/c64281a849ae221f96dc80f33085a3ace587250f/pyserini/output_writer.py#L28
    if format == 'msmarco':
        with open(result_path, 'r') as f:
            for line in f:
                qid, doc_id, rank = line.strip().split('\t')
                if str(qid) != str(doc_id): 
                    pseudo_score = 1 / int(rank)
                    results.setdefault(qid, {})
                    results[qid][doc_id] = pseudo_score
    elif format == 'trec':
        with open(result_path, 'r') as f:
            for line in f:
                qid, _, doc_id, rank, score, _ = line.strip().split()
                if str(qid) != str(doc_id): 
                    score = float(score)
                    results.setdefault(qid, {})
                    results[qid][doc_id] = score
    else:
        raise NotImplementedError
    
    return results

def run(result_path, latency_path: str, index_path: str, format, qrels_path, output_dir, bins: int=10, k_values=[1,3,5,10,100,1000]):
    results = load_results(result_path, format)
    qrels = load_qrels(qrels_path)
    evaluator = EvaluateRetrieval()
    if len(qrels) != len(results):
        missing_queries = set(qrels) - set(results)
        print(f'WARNING: #queries ({len(qrels)}) != |results| ({len(results)}). Queries - results: {set(qrels) - set(results)}')
        for qid in missing_queries:
            results[qid] = {}
            
    # assert len(qrels) == len(results), '#queries should be the same'
    ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, k_values)
    mrr = EvaluateRetrieval.evaluate_custom(qrels, results, k_values, metric='mrr')

    # Get latency info:
    latencies: List[float] = []
    word_lengths: List[int] = []
    batch_sizes: List[int] = []
    with open(latency_path) as f:
        for line in f:
            qid, word_length, latency, batch_size = line.strip().split("\t")
            latencies.append(float(latency))
            word_lengths.append(int(word_length))
            batch_sizes.append(int(batch_size))
    freqs, word_length_bins = np.histogram(word_lengths, bins=bins)
    binned_latencies_avg = bin_and_average(keys=word_lengths, values=latencies, numpy_bins=word_length_bins)
    binned_latencies_std = bin_and_std(keys=word_lengths, values=latencies, numpy_bins=word_length_bins)
    latency_info = {
        "latency": {
            "latency_avg": np.mean(latencies),
            "latency_std": np.std(latencies),
            "query_word_length_avg": np.mean(word_lengths),
            "binned": {
                "word_length_bins": word_length_bins.tolist(),
                "freqs": freqs.tolist(),
                "latencies_avg": binned_latencies_avg,
                "latencies_std": binned_latencies_std
            },
            "batch_size": np.mean(batch_sizes),
            "processor": get_processor_name()
        }
    }

    # Get index info:
    index_size = get_folder_size(index_path)
    index_info = {
        "index_size": index_size
    }

    # Get evaluation scores and save all results:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        metrics = {
            'nDCG': ndcg,
            'MAP': _map,
            'Recall': recall,
            'Precision': precision,
            'mrr': mrr
        }
        metrics.update(latency_info)
        metrics.update(index_info)
        json.dump(metrics, f, indent=4)
    print(f'{__name__}: Done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path')
    parser.add_argument('--latency_path')
    parser.add_argument('--index_path')
    parser.add_argument('--format', choices=['msmarco', 'trec'], help='Format of the retrieval result. The formats are from pyserini.output_writer.py')
    parser.add_argument('--qrels_path', help='Path to the BeIR-format file')
    parser.add_argument('--output_dir')
    parser.add_argument('--bins', type=int, default=10, help="Binning query latencies wrt. how many word-length bins.")
    parser.add_argument('--k_values', nargs='+', type=int, default=[1,2,3,5,10,20,100,1000])
    args = parser.parse_args()
    run(**vars(args))
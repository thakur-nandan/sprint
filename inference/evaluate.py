import argparse
import csv
import json
import os
from typing import Dict
from beir.retrieval.evaluation import EvaluateRetrieval
import crash_ipdb


def load_results(result_path, format) -> Dict[str, Dict[str, float]]:
    results = {}

    # the format is from pyserini's output-format: 
    # https://github.com/castorini/pyserini/blob/c64281a849ae221f96dc80f33085a3ace587250f/pyserini/output_writer.py#L28
    if format == 'msmarco':
        with open(result_path, 'r') as f:
            for line in f:
                qid, doc_id, rank = line.strip().split('\t')
                pseudo_score = 1 / int(rank)
                results.setdefault(qid, {})
                results[qid][doc_id] = pseudo_score
    elif format == 'trec':
        with open(result_path, 'r') as f:
            for line in f:
                qid, _, doc_id, rank, score, _ = line.strip().split()
                score = float(score)
                results.setdefault(qid, {})
                results[qid][doc_id] = score
    else:
        raise NotImplementedError
    
    return results

def load_qrels(qrels_path):
    # adapted from BeIR: 
    # https://github.com/UKPLab/beir/blob/568f4c34fa0be1901e2aaa8479978c9e54a1e377/beir/datasets/data_loader.py#L114
    reader = csv.reader(open(qrels_path, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    next(reader)  # skip header
    
    qrels = {}
    for row in reader:
        query_id, corpus_id, score = row[0], row[1], int(row[2])
        
        if query_id not in qrels:
            qrels[query_id] = {corpus_id: score}
        else:
            qrels[query_id][corpus_id] = score
    
    return qrels

def run(result_path, format, qrels_path, output_dir, k_values=[1,3,5,10,100,1000]):
    results = load_results(result_path, format)
    qrels = load_qrels(qrels_path)
    evaluator = EvaluateRetrieval()
    assert len(qrels) == len(results), '#queries should be the same'
    ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, k_values)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        metrics = {
            'nDCG': ndcg,
            'MAP': _map,
            'Recall': recall,
            'Precision': precision,
        }
        json.dump(metrics, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path')
    parser.add_argument('--format', choices=['msmarco', 'trec'], help='Format of the retrieval result. The formats are from pyserini.output_writer.py')
    parser.add_argument('--qrels_path', help='Path to the BeIR-format file')
    parser.add_argument('--output_dir')
    parser.add_argument('--k_values', nargs='+', type=int, default=[1,3,5,10,100,1000])
    args = parser.parse_args()
    run(**vars(args))
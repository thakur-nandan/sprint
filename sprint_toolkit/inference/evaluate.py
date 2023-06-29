import argparse
import csv
import json
import os
from typing import Dict
from .utils import load_qrels
from beir.retrieval.evaluation import EvaluateRetrieval


def load_results(result_path, format) -> Dict[str, Dict[str, float]]:
    results = {}

    # the format is from pyserini's output-format:
    # https://github.com/castorini/pyserini/blob/c64281a849ae221f96dc80f33085a3ace587250f/pyserini/output_writer.py#L28
    if format == "msmarco":
        with open(result_path, "r") as f:
            for line in f:
                qid, doc_id, rank = line.strip().split("\t")
                if str(qid) != str(doc_id):
                    pseudo_score = 1 / int(rank)
                    results.setdefault(qid, {})
                    results[qid][doc_id] = pseudo_score
    elif format == "trec":
        with open(result_path, "r") as f:
            for line in f:
                qid, _, doc_id, rank, score, _ = line.strip().split()
                if str(qid) != str(doc_id):
                    score = float(score)
                    results.setdefault(qid, {})
                    results[qid][doc_id] = score
    else:
        raise NotImplementedError

    return results


def run(result_path, format, qrels_path, output_dir, k_values=[1, 3, 5, 10, 100, 1000]):
    results = load_results(result_path, format)
    qrels = load_qrels(qrels_path)
    evaluator = EvaluateRetrieval()
    if len(qrels) != len(results):
        missing_queries = set(qrels) - set(results)
        print(
            f"WARNING: #queries ({len(qrels)}) != |results| ({len(results)}). Queries - results: {set(qrels) - set(results)}"
        )
        for qid in missing_queries:
            results[qid] = {}

    # assert len(qrels) == len(results), '#queries should be the same'
    ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, k_values)
    mrr = EvaluateRetrieval.evaluate_custom(qrels, results, k_values, metric="mrr")
    hole = EvaluateRetrieval.evaluate_custom(qrels, results, k_values, metric="hole")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        metrics = {
            "nDCG": ndcg,
            "MAP": _map,
            "Recall": recall,
            "Precision": precision,
            "mrr": mrr,
            "hole": hole,
        }
        json.dump(metrics, f, indent=4)
    print(f"{__name__}: Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path")
    parser.add_argument(
        "--format",
        choices=["msmarco", "trec"],
        help="Format of the retrieval result. The formats are from pyserini.output_writer.py",
    )
    parser.add_argument("--qrels_path", help="Path to the BeIR-format file")
    parser.add_argument("--output_dir")
    parser.add_argument(
        "--k_values", nargs="+", type=int, default=[1, 2, 3, 5, 10, 20, 100, 1000]
    )
    args = parser.parse_args()
    run(**vars(args))

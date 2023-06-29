import csv
import json
from typing import Dict


def load_qrels(qrels_path) -> Dict[str, Dict[str, float]]:
    # adapted from BeIR:
    # https://github.com/beir-cellar/beir/blob/main/beir/datasets/data_loader.py#L114
    reader = csv.reader(
        open(qrels_path, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_MINIMAL
    )
    next(reader)  # skip header

    qrels = {}
    for row in reader:
        query_id, corpus_id, score = row[0], row[1], int(row[2])

        if query_id not in qrels:
            qrels[query_id] = {corpus_id: score}
        else:
            qrels[query_id][corpus_id] = score

    return qrels


def load_queries(queries_path) -> Dict[str, str]:
    # adapted from BEIR:
    # https://github.com/beir-cellar/beir/blob/main/beir/datasets/data_loader.py#L107
    queries = {}
    with open(queries_path, encoding="utf8") as fIn:
        for line in fIn:
            line = json.loads(line)
            queries[line.get("_id")] = line.get("text")

    return queries

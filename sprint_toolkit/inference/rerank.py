from collections import defaultdict
import json
import os
from typing import Dict, List
from numpy import sort
from tqdm.auto import tqdm
from . import encoder_builders
from .evaluate import load_results
import argparse


def load_corpus(corpus_path, format) -> Dict[str, str]:
    print("Loading corpus")
    corpus = {}
    format = format.lower()
    if format == "beir":
        with open(corpus_path, "r") as f:
            for metadata in tqdm(f):
                line = json.loads(metadata)
                doc_id = line["_id"] if "_id" in line else line["id"]
                if "text" in line:
                    text = " ".join([line["title"], line["text"]])
                else:
                    text = " ".join([line["title"], line["contents"]])
                corpus[doc_id] = text
    else:
        raise NotImplementedError
    return corpus


def load_topics(topics_path, format) -> Dict[str, str]:
    format = format.lower()
    queries = {}
    if format == "beir":
        with open(topics_path, "r") as f:
            for line in f:
                line_dict = json.loads(line)
                qid = line_dict["_id"]
                text = line_dict["text"]
                queries[qid] = text
    elif format in ["pyserini", "anserini"]:
        with open(topics_path, "r") as f:
            # 300674	how many years did william bradford serve as governor of plymouth colony?
            for line in f:
                qid, text = line.strip().split("\t")
                queries[qid] = text
    else:
        raise NotImplementedError
    return queries


def save_results(results, output_dir, format):
    format = format.lower()
    if format == "trec":
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "run.tsv"), "w") as f:
            for qid, rels in results.items():
                # 19335 Q0 7267248 1 1381.000000 Anserini
                for i, (doc_id, score) in enumerate(rels.items()):
                    line = (
                        " ".join([qid, "Q0", doc_id, str(i + 1), str(score), "rerank"])
                        + "\n"
                    )
                    f.write(line)
    else:
        raise NotImplementedError


def run(
    topics_path,
    topics_format,
    retrieval_result_path,
    retrieval_result_format,
    corpus_path,
    corpus_format,
    encoder_name,
    ckpt_name,
    output_dir,
    output_format,
    device,
    batch_size,
):
    queries: Dict[str, str] = load_topics(topics_path, topics_format)
    corpus: Dict[str, str] = load_corpus(corpus_path, corpus_format)
    retrieval_results: Dict[str, Dict[str, float]] = load_results(
        retrieval_result_path, retrieval_result_format
    )
    corpus_topk = {
        doc_id: corpus[doc_id]
        for _, rels in retrieval_results.items()
        for doc_id in rels
    }

    query_encoder_builder = encoder_builders.get_builder(
        encoder_name, ckpt_name, "query"
    )
    document_encoder_builder = encoder_builders.get_builder(
        encoder_name, ckpt_name, "document"
    )
    query_encoder = query_encoder_builder(device)
    document_encoder = document_encoder_builder(device)

    print("Building lookup table")
    lookup_table = defaultdict(dict)
    doc_ids = list(corpus_topk.keys())
    documents = list(corpus_topk.values())
    for b in tqdm(range(0, len(doc_ids), batch_size)):
        batch = documents[b : b + batch_size]
        term_weight_dicts: List[Dict[str, float]] = document_encoder.encode(
            batch
        )  # no quantization here
        for doc_id, term_weights in zip(doc_ids[b : b + batch_size], term_weight_dicts):
            for token, weight in term_weights.items():
                lookup_table[token][doc_id] = weight

    print("Doing re-ranking")
    results_rerank = {}
    for qid, query in tqdm(queries.items()):
        if qid not in retrieval_results:
            print(
                f"WARNING: Query {qid} is not in the retrieval results. Has ignored it"
            )
            continue

        term_weights = query_encoder.encode(query)
        rels = defaultdict(float)
        for token, weight_query in term_weights.items():
            for doc_id, weight_doc in lookup_table[token].items():
                if doc_id not in retrieval_results[qid]:  # for re-ranking
                    continue
                rels[doc_id] += weight_query * weight_doc
        rels = dict(sorted(rels.items(), key=lambda x: x[1], reverse=True))
        results_rerank[qid] = rels

    print("Saving results")
    save_results(results_rerank, output_dir, output_format)
    print(f"{__name__}: Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics_path")
    parser.add_argument("--topics_format", default="beir")
    parser.add_argument("--retrieval_result_path")
    parser.add_argument("--retrieval_result_format", default="trec")
    parser.add_argument("--corpus_path")
    parser.add_argument("--corpus_format", default="beir")
    parser.add_argument("--encoder_name")
    parser.add_argument("--ckpt_name")
    parser.add_argument("--output_dir")
    parser.add_argument("--output_format", default="trec")
    parser.add_argument("--device", type=int)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    run(**vars(args))

import json
import shutil
from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import logging
import pathlib, os
import random

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
#### /print debug information to stdout

#### Download scifact.zip dataset and unzip the dataset
dataset = "scifact"
url = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
        dataset
    )
)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), ".")
shutil.rmtree(os.path.join(out_dir, "scifact"))
data_path = util.download_and_unzip(url, out_dir)

corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
random_state = random.Random(42)
corpus_sampled = dict(random_state.sample(list(corpus.items()), k=10))
qrels_sampled = dict(random_state.sample(list(qrels.items()), k=3))
for qid, rels in qrels_sampled.items():
    for did in rels:
        corpus_sampled[did] = corpus[did]
queries_sampled = {qid: queries[qid] for qid, _ in qrels_sampled.items()}


with open(os.path.join(data_path, "corpus.jsonl"), "w") as f:
    for id, line in corpus_sampled.items():
        line.update({"_id": id})
        f.write(json.dumps(line) + "\n")

with open(os.path.join(data_path, "queries.jsonl"), "w") as f:
    for qid, text in queries_sampled.items():
        f.write(json.dumps({"_id": qid, "text": text, "metadata": {}}) + "\n")

with open(os.path.join(data_path, "qrels", "test.tsv"), "w") as f:
    f.write("query-id\tcorpus-id\tscore\n")
    for qid, rels in qrels_sampled.items():
        for did, rel in rels.items():
            f.write(f"{qid}\t{did}\t{rel}\n")

os.remove(os.path.join(data_path, "qrels", "train.tsv"))

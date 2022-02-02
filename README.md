# sparse-retrieval
Train and evaluate all the sparse-retrieval methods in one stop.

## Dependency
This repo is backended by Pyserini, which relies on Java. To make all the things eaiser, we recommend to install all the dependencies via `conda`:
```bash
conda env create -f environment.yml  # The Java/JDK dependency will also be installed by running this
```
This will create a conda environment named `sparse-retrieval`. So if you want other names, please change the `name` argument in [environment.yml](environment.yml).

## Inference
### Quick start
For a quick start, we can go to the [example](examples/inference/distilsplade_max/beir_scifact/all_in_one.sh) for evaluating SPLADE (`distilsplade_max`) on the BeIR/SciFact dataset:
```bash
cd examples/inference/distilsplade_max/beir_scifact
bash all_in_one.sh
```
This will go over the whole pipeline and give the final evaluation results in `beir_scifact-distilsplade_max-quantized/evaluation/metrics.json`:
```bash
cat beir_scifact-distilsplade_max-quantized/evaluation/metrics.json 
# {
#     "nDCG": {
#         "NDCG@1": 0.60333,
#         "NDCG@3": 0.65969,
#         "NDCG@5": 0.67204,
#         "NDCG@10": 0.6925,
#         "NDCG@100": 0.7202,
#         "NDCG@1000": 0.72753
#     },
#     "MAP": {
#         "MAP@1": 0.57217,
#     ...
# }
```

### Step by step
One can also run the above process in 6 separate steps under the [step_by_step](examples/inference/distilsplade_max/beir_scifact/step_by_step) folder:
1. [encode](examples/inference/distilsplade_max/beir_scifact/step_by_step/1.encode.beir_scifact-distilsplade_max-float.sh): Encode documents into term weights by multiprocessing on mutliple GPUs;
2. [quantize](examples/inference/distilsplade_max/beir_scifact/step_by_step/2.quantize.beir_scifact-distilsplade_max-2digits.sh): Quantize the document term weights into integers (can be scaped);
3. [index](examples/inference/distilsplade_max/beir_scifact/step_by_step/3.index.beir_scifact-distilsplade_max-2digits.sh): Index the term weights in to Lucene index (backended by Pyserini);
4. [reformat](examples/inference/distilsplade_max/beir_scifact/step_by_step/4.reformat_query.beir_scifact.sh): Reformat the queries file (e.g. the ones from BeIR) into the Pyserini format;
5. [search](examples/inference/distilsplade_max/beir_scifact/step_by_step/5.search.beir_scifact-distilsplade_max-2digits.sh): Retrieve the relevant documents (backended by Pyserini);
6. [evaluate](examples/inference/distilsplade_max/beir_scifact/step_by_step/6.evaluate.beir_scifact-distilsplade_max-2digits.sh): Evaluate the results against a certain labeled data, e.g.the qrels used in BeIR (backended by BeIR)

Currently it supports methods:
- uniCOIL
- SPLADE: Go to [examples/inference/distilsplade_max/beir_scifact](examples/inference/distilsplade_max/beir_scifact) for fast reproducing `distilsplade_max` on SciFact;

Currently it supports data (by downloading automatically):
- BeIR

Other models and data (formats) will be added.

## Training
Will be added.



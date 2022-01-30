# sparse-retrieval
Train and evaluate all the sparse-retrieval methods in one stop.

## Dependency
This repo is backended by Pyserini, which relies on Java. To make all the things eaiser, we recommend to install all the dependencies via `conda`:
```bash
conda env create -f environment.yml  # The Java/JDK dependency will also be installed by running this
```
This will create a conda environment named `sparse-retrieval`. So if you want other names, please change the `name` argument in [environment.yml](environment.yml).

## Inference

The inference in this repo includes 6 stages/steps: (links to the corresponding example script)
1. [encode](examples/inference/unicoil-noexp/beir_scifact/1.encode.beir_scifact-unicoil_noexp-float.sh): Encode documents into term weights by multiprocessing on mutliple GPUs;
2. [quantize](examples/inference/unicoil-noexp/beir_scifact/2.quantize.beir_scifact-unicoil_noexp-b8.sh): Quantize the document term weights into integers (can be scaped);
3. [index](examples/inference/unicoil-noexp/beir_scifact/3.index.beir_scifact-unicoil_noexp-b8.sh): Index the term weights in to Lucene index (backended by Pyserini);
4. [reformat](examples/inference/unicoil-noexp/beir_scifact/4.reformat_query.beir_scifact.sh): Reformat the queries file (e.g. the ones from BeIR) into the Pyserini format;
5. [search](examples/inference/unicoil-noexp/beir_scifact/5.search.beir_scifact-unicoil_noexp-b8.sh): Retrieve the relevant documents (backended by Pyserini);
6. [evaluate](examples/inference/unicoil-noexp/beir_scifact/6.evaluate.beir_scifact-unicoil_noexp-b8.sh): Evaluate the results against a certain labeled data, e.g.the qrels used in BeIR (backended by BeIR)

One can directly go to the example for evaluating uniCOIL (w/o document expansion) on `BeIR/scifact`: [examples/inference/unicoil-noexp/beir_scifact](examples/inference/unicoil-noexp/beir_scifact) and run the example scripts one by one.

NOTICE: Please change the `$SPARSE_RETRIEVAL_HOME` in the scripts into your local path to the repo, if needed.

Currently it supports methods:
- uniCOIL
- SPLADE: Go to [examples/inference/distilsplade_max/beir_scifact](examples/inference/distilsplade_max/beir_scifact) for fast reproducing `distilsplade_max` on SciFact;

Currently it supports data (by downloading automatically):
- BeIR

Other models and data (formats) will be added.

## Training
Will be added.



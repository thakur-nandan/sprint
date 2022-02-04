# sparse-retrieval
Train and evaluate all the sparse-retrieval methods in one stop.

## Dependency and installation
This repo is backended by Pyserini, which relies on Java. To make all the things eaiser, we recommend to install all the dependencies via `conda`:
```bash
conda env create -f environment.yml  # The Java/JDK dependency will also be installed by running this
```
This will create a conda environment named `sparse-retrieval`. So if you want other names, please change the `name` argument in [environment.yml](environment.yml).

To install this repo, just go into the repo and do:  (This is required to run the examples)
```
pip install -e .
```


## Inference
### Quick start
For a quick start, we can go to the [example](examples/inference/distilsplade_max/beir_scifact/all_in_one.sh) for evaluating SPLADE (`distilsplade_max`) on the BeIR/SciFact dataset:
```bash
cd examples/inference/distilsplade_max/beir_scifact
bash all_in_one.sh
```
This will go over the whole pipeline and give the final evaluation results in `beir_scifact-distilsplade_max-quantized/evaluation/metrics.json`:

<details>
  <summary>Results: distilsplade_max on BeIR/SciFact</summary>
  
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
</details>

Or if you like running python directly, just run the code snippet below for evaluating `castorini/unicoil-noexp-msmarco-passage` on `BeIR/SciFact`:
```python
from sparse_retrieval.inference import aio


if __name__ == '__main__':  # aio.run can only be called within __main__
    aio.run(
        encoder_name='unicoil',
        ckpt_name='castorini/unicoil-noexp-msmarco-passage',
        data_name='beir/scifact',
        gpus=[0, 1],
        output_dir='beir_scifact-unicoil_noexp',
        do_quantization=True,
        quantization_method='range-nbits',  # So the doc term weights will be quantized by `(term_weights / 5) * (2 ** 8)`
        original_score_range=5,
        quantization_nbits=8,
        original_query_format='beir',
        topic_split='test'
    )
    # You would get "NDCG@10": 0.68563
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

### Custom encoders
To add a custom encoder, one can refer to the example [examples/inference/custom_encoder/beir_scifact](examples/inference/custom_encoder/beir_scifact), where `distilsplade_max` is evaluated on `BeIR/SciFact` **with stopwords filtered out**.

In detail, one just needs to define your custom encoder class and write a new encoder builder function:
```python
from typing import Dict, List
from pyserini.encode import QueryEncoder, DocumentEncoder

class CustomQueryEncoder(QueryEncoder):

    def encode(self, text, **kwargs) -> Dict[str, float]:
        # Just an example:
        terms = text.split()
        term_weights = {term: 1 for term in terms}
        return term_weights  # Dict object, where keys/values are terms/term scores, resp.

class CustomDocumentEncoder(DocumentEncoder):

    def encode(self, texts, **kwargs) -> List[Dict[str, float]]:
        # Just an example:
        term_weights_batch = []
        for text in texts:
            terms = text.split()
            term_weights = {term: 1 for term in terms}
            term_weights_batch.append(term_weights)
        return term_weights_batch 

def custom_encoder_builder(ckpt_name, etype, device='cpu'):
    if etype == 'query':
        return CustomQueryEncoder(ckpt_name, device=device)        
    elif etype == 'document':
        return CustomDocumentEncoder(ckpt_name, device=device)
    else:
        raise ValueError
```
Then register `custom_encoder_builder` with `sparse_retrieval.inference.encoder_builders.register` before usage:
```python
from sparse_retrieval.inference.encoder_builders import register

register('custom_encoder_builder', custom_encoder_builder)
```

## Training
Will be added.



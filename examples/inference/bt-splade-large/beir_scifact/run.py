"""
This script shows how to evaluate BT-SPLADE-L model on SciFact dataset.
The BT-SPLADE-L model contains two encoders: one for queries and one for documents.
The query encoder is a Tiny BERT model, and the document encoder is a DistilBERT model. 
BT-SPLADE-L query encoder: https://huggingface.co/naver/efficient-splade-VI-BT-large-query
BT-SPLADE-L document encoder: https://huggingface.co/naver/efficient-splade-VI-BT-large-doc
For more details, refer to (Lassance et. al. 2022): https://dl.acm.org/doi/10.1145/3477495.3531833

Parameters:
You can add multiple GPUs in the `--gpus` parameter for faster inference. 
Add `beir_` before the dataset name in `--data_name` parameter. 
Dataset will get downloaded in your current path (\datasets) if not present.
Add model checkpoints (query, document) in `--ckpt_name` parameter.
Add `--do_quantization` parameter to enable quantization.
Add `--quantization_method` parameter to specify ndigits-round and `--ndigits` = 2 for rounding off by x100.
"""

from sprint_toolkit.inference import aio


if __name__ == '__main__':  # aio.run can only be called within __main__

    aio.run(
        encoder_name='splade',
        ckpt_name=['naver/efficient-splade-VI-BT-large-query', 'naver/efficient-splade-VI-BT-large-doc'],
        data_name='beir/scifact',
        gpus=[0],
        output_dir='beir_scifact-bt-splade-l',
        do_quantization=True,
        quantization_method='ndigits-round',
        ndigits=2,
        original_query_format='beir',
        topic_split='test'
    )

# You should get the following results on SCIFACT dataset with BT-SPLADE-L model:
# {
#     "nDCG": {
#         "NDCG@1": 0.56667,
#         "NDCG@2": 0.61714,
#         "NDCG@3": 0.63374,
#         "NDCG@5": 0.64956,
#         "NDCG@10": 0.67369,
#         "NDCG@20": 0.68854,
#         "NDCG@100": 0.70123,
#         "NDCG@1000": 0.71067
#     },
#     "MAP": {
#         "MAP@1": 0.54317,
#         "MAP@2": 0.59356,
#         "MAP@3": 0.6093,
#         "MAP@5": 0.62055,
#         "MAP@10": 0.63238,
#         "MAP@20": 0.63704,
#         "MAP@100": 0.63933,
#         "MAP@1000": 0.63968
#     },
#     "Recall": {
#         "Recall@1": 0.54317,
#         "Recall@2": 0.63661,
#         "Recall@3": 0.67772,
#         "Recall@5": 0.718,
#         "Recall@10": 0.78656,
#         "Recall@20": 0.84267,
#         "Recall@100": 0.90367,
#         "Recall@1000": 0.97833
#     },
#     "Precision": {
#         "P@1": 0.56667,
#         "P@2": 0.34167,
#         "P@3": 0.24556,
#         "P@5": 0.15733,
#         "P@10": 0.088,
#         "P@20": 0.04733,
#         "P@100": 0.01027,
#         "P@1000": 0.00111
#     },
#     "mrr": {
#         "MRR@1": 0.56667,
#         "MRR@2": 0.61333,
#         "MRR@3": 0.62667,
#         "MRR@5": 0.6355,
#         "MRR@10": 0.6447,
#         "MRR@20": 0.64812,
#         "MRR@100": 0.64971,
#         "MRR@1000": 0.65002
#     },
#     "hole": {
#         "Hole@1": 0.41,
#         "Hole@2": 0.61833,
#         "Hole@3": 0.71556,
#         "Hole@5": 0.80133,
#         "Hole@10": 0.85733,
#         "Hole@20": 0.89733,
#         "Hole@100": 0.9303,
#         "Hole@1000": 0.94051
#     }
# }
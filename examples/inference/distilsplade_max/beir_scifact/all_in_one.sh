# This script shows how to evaluate SPLADEv2 model on SciFact dataset.
# The SPLADEv2 model contains one encoder for both queries and documents.
# The query and document encoder is a DistilBERT model. 
# SPLADEv2: https://huggingface.co/naver/splade_v2_distil
# For more details, refer to (Formal et. al. 2021): https://arxiv.org/abs/2109.10086

# Parameters:
# You can add multiple GPUs in the `--gpus` parameter for faster inference. 
# Add `beir_` before the dataset name in `--data_name` parameter. 
# Dataset will get downloaded in your current path (\datasets) if not present.
# Add model checkpoints (query, document) in `--ckpt_name` parameter.
# Add `--do_quantization` parameter to enable quantization.
# Add `--quantization_method` parameter to specify ndigits-round and `--ndigits` = 2 for rounding off by x100.


python -m sprint_toolkit.inference.aio \
    --encoder_name splade \
    --ckpt_name naver/splade_v2_distil \
    --data_name beir_scifact \
    --gpus 0 \
    --output_dir beir_scifact-distilsplade_max \
    --do_quantization \
    --quantization_method ndigits-round \
    --ndigits 2 \
    --original_query_format beir \
    --topic_split test

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
#         "MAP@3": 0.63391,
#         "MAP@5": 0.64403,
#         "MAP@10": 0.65444,
#         "MAP@100": 0.66071,
#         "MAP@1000": 0.66096
#     },
#     "Recall": {
#         "Recall@1": 0.57217,
#         "Recall@3": 0.70172,
#         "Recall@5": 0.73461,
#         "Recall@10": 0.79122,
#         "Recall@100": 0.92033,
#         "Recall@1000": 0.98
#     },
#     "Precision": {
#         "P@1": 0.60333,
#         "P@3": 0.25444,
#         "P@5": 0.16267,
#         "P@10": 0.08967,
#         "P@100": 0.01043,
#         "P@1000": 0.00111
#     },
#     "mrr": {
#         "MRR@1": 0.60333,
#         "MRR@3": 0.65722,
#         "MRR@5": 0.66306,
#         "MRR@10": 0.67052,
#         "MRR@100": 0.67503,
#         "MRR@1000": 0.67524
#     }
# }

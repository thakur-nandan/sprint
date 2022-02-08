# Please make sure you have installed the repo

# Notice that it should be MS MARCO dev split
nohup python -m sparse_retrieval.inference.aio \
    --encoder_name sparta \
    --ckpt_name "BeIR/sparta-msmarco-distilbert-base-v1" \
    --data_name beir_msmarco \
    --gpus 5 6 9 10 15 \
    --output_dir beir_msmarco-sparta \
    --do_quantization \
    --quantization_method ndigits-round \
    --ndigits 2 \
    --original_query_format beir \
    --topic_split dev \
    > all_in_one.log &

# Results: 0.1 nDCG@10 higher than reported in the BeIR paper. This might be due to the quantization method and/or evaluation variance
# {
#     "nDCG": {
#         "NDCG@1": 0.18983,
#         "NDCG@3": 0.2812,
#         "NDCG@5": 0.31628,
#         "NDCG@10": 0.35236,
#         "NDCG@100": 0.40644,
#         "NDCG@1000": 0.42283
#     },
#     "MAP": {
#         "MAP@1": 0.18473,
#         "MAP@3": 0.2564,
#         "MAP@5": 0.27612,
#         "MAP@10": 0.29119,
#         "MAP@100": 0.30221,
#         "MAP@1000": 0.30285
#     },
#     "Recall": {
#         "Recall@1": 0.18473,
#         "Recall@3": 0.34743,
#         "Recall@5": 0.43175,
#         "Recall@10": 0.54177,
#         "Recall@100": 0.79355,
#         "Recall@1000": 0.92016
#     },
#     "Precision": {
#         "P@1": 0.18983,
#         "P@3": 0.1202,
#         "P@5": 0.08983,
#         "P@10": 0.0565,
#         "P@100": 0.00837,
#         "P@1000": 0.00098
#     },
#     "mrr": {
#         "MRR@1": 0.18997,
#         "MRR@3": 0.26091,
#         "MRR@5": 0.28056,
#         "MRR@10": 0.29527,
#         "MRR@100": 0.30607,
#         "MRR@1000": 0.30666
#     }
# }
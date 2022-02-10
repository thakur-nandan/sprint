# Please make sure you have installed the repo

# Notice that it should be MS MARCO dev split
nohup python -m sparse_retrieval.inference.aio \
    --encoder_name unicoil \
    --ckpt_name "castorini/unicoil-noexp-msmarco-passage" \
    --data_name beir_msmarco \
    --gpus 9 10 13 14 \
    --output_dir beir_msmarco-unicoil-noexp \
    --do_quantization \
    --quantization_method range-nbits \
    --original_score_range 5 \
    --quantization_nbits 8 \
    --original_query_format beir \
    --topic_split dev \
    > all_in_one.log &


# {
#     "nDCG": {
#         "NDCG@1": 0.2063,
#         "NDCG@3": 0.30071,
#         "NDCG@5": 0.33633,
#         "NDCG@10": 0.37158,
#         "NDCG@100": 0.42457,
#         "NDCG@1000": 0.43953
#     },
#     "MAP": {
#         "MAP@1": 0.20133,
#         "MAP@3": 0.27523,
#         "MAP@5": 0.29516,
#         "MAP@10": 0.31004,
#         "MAP@100": 0.32089,
#         "MAP@1000": 0.32146
#     },
#     "Recall": {
#         "Recall@1": 0.20133,
#         "Recall@3": 0.36897,
#         "Recall@5": 0.45457,
#         "Recall@10": 0.56108,
#         "Recall@100": 0.80735,
#         "Recall@1000": 0.92375
#     },
#     "Precision": {
#         "P@1": 0.2063,
#         "P@3": 0.12746,
#         "P@5": 0.09433,
#         "P@10": 0.0585,
#         "P@100": 0.00851,
#         "P@1000": 0.00098
#     },
#     "mrr": {
#         "MRR@1": 0.20602,
#         "MRR@3": 0.28064,
#         "MRR@5": 0.30045,
#         "MRR@10": 0.31518,
#         "MRR@100": 0.32559,
#         "MRR@1000": 0.3261
#     }
# }
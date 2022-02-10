export SPARSE_RETRIEVAL_HOME=../../../../  # Path to the repo, please change it accordingly
export PYTHONPATH=$SPARSE_RETRIEVAL_HOME:"${PYTHONPATH}"

nohup python -m sparse_retrieval.inference.aio \
    --encoder_name unicoil \
    --ckpt_name castorini/unicoil-noexp-msmarco-passage \
    --data_name beir_scifact \
    --gpus 6 11 \
    --output_dir beir_scifact-unicoil_noexp \
    --do_quantization \
    --quantization_method range-nbits \
    --original_score_range 5 \
    --quantization_nbits 8 \
    --original_query_format beir \
    --topic_split test \
    > all_in_one.log &

# {
#     "nDCG": {
#         "NDCG@1": 0.58333,
#         "NDCG@3": 0.64131,
#         "NDCG@5": 0.66548,
#         "NDCG@10": 0.68563,
#         "NDCG@100": 0.70949,
#         "NDCG@1000": 0.71736
#     },
#     "MAP": {
#         "MAP@1": 0.55261,
#         "MAP@3": 0.6178,
#         "MAP@5": 0.633,
#         "MAP@10": 0.64296,
#         "MAP@100": 0.64808,
#         "MAP@1000": 0.64837
#     },
#     "Recall": {
#         "Recall@1": 0.55261,
#         "Recall@3": 0.6865,
#         "Recall@5": 0.74517,
#         "Recall@10": 0.80278,
#         "Recall@100": 0.912,
#         "Recall@1000": 0.97433
#     },
#     "Precision": {
#         "P@1": 0.58333,
#         "P@3": 0.24667,
#         "P@5": 0.16467,
#         "P@10": 0.09,
#         "P@100": 0.01033,
#         "P@1000": 0.0011
#     },
#     "mrr": {
#         "MRR@1": 0.58333,
#         "MRR@3": 0.63667,
#         "MRR@5": 0.64933,
#         "MRR@10": 0.65673,
#         "MRR@100": 0.6608,
#         "MRR@1000": 0.66105
#     }
# }
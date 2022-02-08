# Please make sure you have installed the repo

nohup python -m sparse_retrieval.inference.aio \
    --encoder_name sparta \
    --ckpt_name "BeIR/sparta-msmarco-distilbert-base-v1" \
    --data_name beir_scifact \
    --gpus 5 6 \
    --output_dir beir_scifact-sparta \
    --do_quantization \
    --quantization_method ndigits-round \
    --ndigits 2 \
    --original_query_format beir \
    --topic_split test \
    > all_in_one.log &

# Results:
# {
#     "nDCG": {
#         "NDCG@1": 0.50333,
#         "NDCG@3": 0.55449,
#         "NDCG@5": 0.56897,
#         "NDCG@10": 0.59334,
#         "NDCG@100": 0.62699,
#         "NDCG@1000": 0.64186
#     },
#     "MAP": {
#         "MAP@1": 0.48611,
#         "MAP@3": 0.53407,
#         "MAP@5": 0.54468,
#         "MAP@10": 0.55604,
#         "MAP@100": 0.56299,
#         "MAP@1000": 0.56365
#     },
#     "Recall": {
#         "Recall@1": 0.48611,
#         "Recall@3": 0.59306,
#         "Recall@5": 0.62733,
#         "Recall@10": 0.69706,
#         "Recall@100": 0.855,
#         "Recall@1000": 0.966
#     },
#     "Precision": {
#         "P@1": 0.50333,
#         "P@3": 0.21222,
#         "P@5": 0.138,
#         "P@10": 0.078,
#         "P@100": 0.00967,
#         "P@1000": 0.00109
#     },
#     "mrr": {
#         "MRR@1": 0.50333,
#         "MRR@3": 0.55,
#         "MRR@5": 0.5575,
#         "MRR@10": 0.56677,
#         "MRR@100": 0.57241,
#         "MRR@1000": 0.57298
#     }
# }
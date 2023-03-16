if [ ! -d "deepimpact-bert-base" ]; then
    wget https://public.ukp.informatik.tu-darmstadt.de/kwang/sparse-retrieval/checkpoints/deepimpact-bert-base.zip
    unzip deepimpact-bert-base.zip
fi


# Please make sure you have installed the repo
# Set `original_score_range = -1` for automatic computing score range,
# which is refered to https://github.com/DI4IR/SIGIR2021/blob/a6b1ee4efaba7d0de75501f2f05a4b9353cdb673/scripts/quantize.py#L27
nohup python -m sprint.inference.aio \
    --encoder_name deepimpact \
    --ckpt_name deepimpact-bert-base \
    --data_name beir/scifact \
    --gpus 0 \
    --output_dir beir_scifact-deepimpact \
    --do_quantization \
    --quantization_method range-nbits \
    --original_score_range -1 \
    --quantization_nbits 8 \
    --original_query_format beir \
    --topic_split test \
    > all_in_one.log &

# {
#     "nDCG": {
#         "NDCG@1": 0.53667,
#         "NDCG@2": 0.57371,
#         "NDCG@3": 0.59072,
#         "NDCG@5": 0.61832,
#         "NDCG@10": 0.63439,
#         "NDCG@20": 0.64589,
#         "NDCG@100": 0.66503,
#         "NDCG@1000": 0.67422
#     },
#     "MAP": {
#         "MAP@1": 0.51667,
#         "MAP@2": 0.55625,
#         "MAP@3": 0.57051,
#         "MAP@5": 0.58705,
#         "MAP@10": 0.59487,
#         "MAP@20": 0.59806,
#         "MAP@100": 0.60125,
#         "MAP@1000": 0.60162
#     },
#     "Recall": {
#         "Recall@1": 0.51667,
#         "Recall@2": 0.58806,
#         "Recall@3": 0.62861,
#         "Recall@5": 0.69678,
#         "Recall@10": 0.74133,
#         "Recall@20": 0.78517,
#         "Recall@100": 0.88144,
#         "Recall@1000": 0.95222
#     },
#     "Precision": {
#         "P@1": 0.53667,
#         "P@2": 0.315,
#         "P@3": 0.22556,
#         "P@5": 0.15133,
#         "P@10": 0.082,
#         "P@20": 0.04367,
#         "P@100": 0.00997,
#         "P@1000": 0.00108
#     },
#     "mrr": {
#         "MRR@1": 0.53667,
#         "MRR@2": 0.575,
#         "MRR@3": 0.58611,
#         "MRR@5": 0.60244,
#         "MRR@10": 0.60809,
#         "MRR@20": 0.61137,
#         "MRR@100": 0.6136,
#         "MRR@1000": 0.6139
#     }
# }
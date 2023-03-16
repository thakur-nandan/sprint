if [ ! -d "deepimpact-bert-base" ]; then
    wget https://public.ukp.informatik.tu-darmstadt.de/kwang/sparse-retrieval/checkpoints/deepimpact-bert-base.zip
    unzip deepimpact-bert-base.zip
fi

if [ ! -d "datasets/msmarco-deepimpact" ]; then
    mkdir -p datasets
    cd datasets
    wget https://public.ukp.informatik.tu-darmstadt.de/kwang/sparse-retrieval/data/msmarco-deepimpact.zip
    unzip msmarco-deepimpact.zip
    cd ..
fi

# Please make sure you have installed the repo
# Set `original_score_range = -1` for automatic computing score range,
# which is refered to https://github.com/DI4IR/SIGIR2021/blob/a6b1ee4efaba7d0de75501f2f05a4b9353cdb673/scripts/quantize.py#L27
nohup python -m sprint.inference.aio \
    --encoder_name deepimpact \
    --ckpt_name deepimpact-bert-base \
    --data_name beir \
    --data_dir datasets/msmarco-deepimpact \
    --gpus 0 \
    --output_dir msmarco-deepimpact-deepimpact \
    --do_quantization \
    --quantization_method range-nbits \
    --original_score_range -1 \
    --quantization_nbits 8 \
    --original_query_format beir \
    --topic_split dev \
    > all_in_one.log &

# TODO: change the quantization into range-nbits: https://github.com/DI4IR/SIGIR2021/blob/main/scripts/quantize.py

# {
#     "nDCG": {
#         "NDCG@1": 0.21289,
#         "NDCG@3": 0.31639,
#         "NDCG@5": 0.35396,
#         "NDCG@10": 0.38964,
#         "NDCG@100": 0.44418,
#         "NDCG@1000": 0.45761
#     },
#     "MAP": {
#         "MAP@1": 0.20681,
#         "MAP@3": 0.28829,
#         "MAP@5": 0.30941,
#         "MAP@10": 0.32447,
#         "MAP@100": 0.33563,
#         "MAP@1000": 0.33613
#     },
#     "Recall": {
#         "Recall@1": 0.20681,
#         "Recall@3": 0.39124,
#         "Recall@5": 0.48132,
#         "Recall@10": 0.58948,
#         "Recall@100": 0.84342,
#         "Recall@1000": 0.94759
#     },
#     "Precision": {
#         "P@1": 0.21289,
#         "P@3": 0.13539,
#         "P@5": 0.10011,
#         "P@10": 0.06146,
#         "P@100": 0.0089,
#         "P@1000": 0.00101
#     },
#     "mrr": {
#         "MRR@1": 0.21175,
#         "MRR@3": 0.29238,
#         "MRR@5": 0.31368,
#         "MRR@10": 0.32835,
#         "MRR@100": 0.33909,
#         "MRR@1000": 0.33955
#     }
# }
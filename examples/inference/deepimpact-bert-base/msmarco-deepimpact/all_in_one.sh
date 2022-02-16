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

nohup python -m sparse_retrieval.inference.aio \
    --encoder_name deepimpact \
    --ckpt_name deepimpact-bert-base \
    --data_name beir \
    --data_dir datasets/msmarco-deepimpact \
    --gpus 6 8 9 11 \
    --output_dir msmarco-deepimpact-deepimpact \
    --do_quantization \
    --quantization_method ndigits-round \
    --ndigits 3 \
    --original_query_format beir \
    --topic_split dev \
    >> all_in_one.log &

# TODO: change the quantization into range-nbits: https://github.com/DI4IR/SIGIR2021/blob/main/scripts/quantize.py

# {
#     "nDCG": {
#         "NDCG@1": 0.21132,
#         "NDCG@3": 0.31453,
#         "NDCG@5": 0.35253,
#         "NDCG@10": 0.38819,
#         "NDCG@100": 0.44286,
#         "NDCG@1000": 0.45637
#     },
#     "MAP": {
#         "MAP@1": 0.2053,
#         "MAP@3": 0.2866,
#         "MAP@5": 0.30793,
#         "MAP@10": 0.32296,
#         "MAP@100": 0.33412,
#         "MAP@1000": 0.33462
#     },
#     "Recall": {
#         "Recall@1": 0.2053,
#         "Recall@3": 0.38899,
#         "Recall@5": 0.48017,
#         "Recall@10": 0.58846,
#         "Recall@100": 0.84321,
#         "Recall@1000": 0.94802
#     },
#     "Precision": {
#         "P@1": 0.21132,
#         "P@3": 0.13457,
#         "P@5": 0.09989,
#         "P@10": 0.06132,
#         "P@100": 0.0089,
#         "P@1000": 0.00101
#     },
#     "mrr": {
#         "MRR@1": 0.21132,
#         "MRR@3": 0.2931,
#         "MRR@5": 0.3141,
#         "MRR@10": 0.32888,
#         "MRR@100": 0.33951,
#         "MRR@1000": 0.33997
#     }
# }
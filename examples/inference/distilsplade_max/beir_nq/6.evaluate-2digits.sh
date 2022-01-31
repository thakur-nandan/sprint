export SPARSE_RETRIEVAL_HOME=../../../../  # Path to the repo, please change it accordingly
export PYTHONPATH=$SPARSE_RETRIEVAL_HOME:"${PYTHONPATH}"

export stage=evaluate

export encoder_name='splade'
export encoder_ckpt_name='distilsplade_max'  # Here we use noexp model (i.e. no document expansion), since the documents are not expanded
export data_name='beir_nq'  # beir data can be downloaded automatically
export quantization=2digits  # The current encoding stage will output the original float weights without quantization

export long_idenitifer=$data_name-$encoder_ckpt_name-$quantization

export output_dir=$long_idenitifer/index
export log_name=$stage.$long_idenitifer.log
export index_dir=$long_idenitifer/index

export format=trec  # The output format used in the last stage (i.e. search)
export result_path=$long_idenitifer/search/$format-format/run.tsv

export data_dir=datasets/beir/nq
export qrels_path=$data_dir/qrels/test.tsv
export output_dir=$long_idenitifer/evaluation
export k_values="1 3 5 10 100 1000"  # The cutoff values used in evlauation

python -m inference.$stage \
    --result_path $result_path \
    --format $format \
    --qrels_path $qrels_path \
    --output_dir $output_dir \
    --k_values $k_values

# ** Actually this stage will not output any logs **

# Metrics are shown below for checking whether everything has gone smoothly:
# (So you would get this in sparse-retrieval/examples/inference/distilsplade_max/beir_nq/beir_nq-distilsplade_max-2digits/evaluation/metrics.json)
# {
#     "nDCG": {
#         "NDCG@1": 0.33633,
#         "NDCG@3": 0.44048,
#         "NDCG@5": 0.48112,
#         "NDCG@10": 0.52084,
#         "NDCG@100": 0.56725,
#         "NDCG@1000": 0.57438
#     },
#     "MAP": {
#         "MAP@1": 0.30048,
#         "MAP@3": 0.40259,
#         "MAP@5": 0.42704,
#         "MAP@10": 0.44493,
#         "MAP@100": 0.45578,
#         "MAP@1000": 0.45609
#     },
#     "Recall": {
#         "Recall@1": 0.30048,
#         "Recall@3": 0.51854,
#         "Recall@5": 0.61208,
#         "Recall@10": 0.72755,
#         "Recall@100": 0.93074,
#         "Recall@1000": 0.9831
#     },
#     "Precision": {
#         "P@1": 0.33633,
#         "P@3": 0.19921,
#         "P@5": 0.14328,
#         "P@10": 0.08615,
#         "P@100": 0.01125,
#         "P@1000": 0.00119
#     },
#     "mrr": {
#         "MRR@1": 0.33662,
#         "MRR@3": 0.43472,
#         "MRR@5": 0.45509,
#         "MRR@10": 0.47026,
#         "MRR@100": 0.47824,
#         "MRR@1000": 0.47846
#     }
# }
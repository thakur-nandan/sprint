export SPARSE_RETRIEVAL_HOME=../../../../  # Path to the repo, please change it accordingly
export PYTHONPATH=$SPARSE_RETRIEVAL_HOME:"${PYTHONPATH}"

export stage=evaluate

export encoder_name='splade'
export encoder_ckpt_name='distilsplade_max'  # Here we use noexp model (i.e. no document expansion), since the documents are not expanded
export data_name='beir_scifact'  # beir data can be downloaded automatically
export quantization=float  # The current encoding stage will output the original float weights without quantization

export long_idenitifer=$data_name-$encoder_ckpt_name-$quantization

export output_dir=$long_idenitifer/index
export log_name=$stage.$long_idenitifer.log
export index_dir=$long_idenitifer/index

export format=trec  # The output format used in the last stage (i.e. search)
export result_path=$long_idenitifer/search/$format-format/run.tsv

export data_dir=datasets/beir/scifact
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
# (So you would get this in sparse-retrieval/examples/inference/distilsplade_max/beir_scifact/beir_scifact-distilsplade_max-float/evaluation/metrics.json)
# It seems that without quantization the performance will be lower (69.3 nDCG@10 -> 66.1 nDCG@10)
# {
#     "nDCG": {
#         "NDCG@1": 0.55333,
#         "NDCG@3": 0.62692,
#         "NDCG@5": 0.63786,
#         "NDCG@10": 0.66149,
#         "NDCG@100": 0.68989,
#         "NDCG@1000": 0.69799
#     },
#     "MAP": {
#         "MAP@1": 0.52217,
#         "MAP@3": 0.59796,
#         "MAP@5": 0.60635,
#         "MAP@10": 0.61825,
#         "MAP@100": 0.62472,
#         "MAP@1000": 0.62507
#     },
#     "Recall": {
#         "Recall@1": 0.52217,
#         "Recall@3": 0.67794,
#         "Recall@5": 0.70906,
#         "Recall@10": 0.77511,
#         "Recall@100": 0.90533,
#         "Recall@1000": 0.96667
#     },
#     "Precision": {
#         "P@1": 0.55333,
#         "P@3": 0.24667,
#         "P@5": 0.156,
#         "P@10": 0.08767,
#         "P@100": 0.01027,
#         "P@1000": 0.0011
#     },
#     "mrr": {
#         "MRR@1": 0.55333,
#         "MRR@3": 0.62,
#         "MRR@5": 0.627,
#         "MRR@10": 0.63531,
#         "MRR@100": 0.64023,
#         "MRR@1000": 0.64053
#     }
# }
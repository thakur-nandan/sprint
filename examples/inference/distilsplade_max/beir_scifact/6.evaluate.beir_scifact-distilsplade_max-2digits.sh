export SPARSE_RETRIEVAL_HOME=../../../../  # Path to the repo, please change it accordingly
export PYTHONPATH=$SPARSE_RETRIEVAL_HOME:"${PYTHONPATH}"

export stage=evaluate

export encoder_name='splade'
export encoder_ckpt_name='distilsplade_max'  # Here we use noexp model (i.e. no document expansion), since the documents are not expanded
export data_name='beir_scifact'  # beir data can be downloaded automatically
export quantization=2digits  # The current encoding stage will output the original float weights without quantization

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
# (So you would get this in sparse-retrieval/examples/inference/distilsplade_max/beir_scifact/beir_scifact-distilsplade_max-2digits/evaluation/metrics.json)
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
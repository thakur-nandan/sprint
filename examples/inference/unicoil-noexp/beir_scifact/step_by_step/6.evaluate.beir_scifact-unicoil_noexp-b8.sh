export SPRINT_HOME=../../../../  # Path to the repo, please change it accordingly
export PYTHONPATH=$SPRINT_HOME:"${PYTHONPATH}"

export stage=evaluate

export encoder_name='unicoil'
export encoder_ckpt_name='unicoil_noexp'
export data_name='beir_scifact'
export quantization='b8'

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
# (So you would get this in sparse-retrieval/examples/inference/unicoil-noexp/beir_scifact/beir_scifact-unicoil_noexp-b8/evaluation/metrics.json)
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
#     }
# }

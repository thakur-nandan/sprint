# Step 6: Evaluate the search results using BEIR test qrels
export SPRINT_HOME=../../../../../  # Path to the repo, please change it accordingly
export PYTHONPATH=$SPRINT_HOME:"${PYTHONPATH}"

export stage=evaluate

export encoder_name='splade'
export encoder_ckpt_name='bt-splade-l'
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

python -m sprint_toolkit.inference.$stage \
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
#         "NDCG@1": 0.56667,
#         "NDCG@3": 0.63374,
#         "NDCG@5": 0.64956,
#         "NDCG@10": 0.67369,
#         "NDCG@100": 0.70123,
#         "NDCG@1000": 0.71067
#     },
#     "MAP": {
#         "MAP@1": 0.54317,
#         "MAP@3": 0.6093,
#         "MAP@5": 0.62055,
#         "MAP@10": 0.63238,
#         "MAP@100": 0.63933,
#         "MAP@1000": 0.63968
#     },
#     "Recall": {
#         "Recall@1": 0.54317,
#         "Recall@3": 0.67772,
#         "Recall@5": 0.718,
#         "Recall@10": 0.78656,
#         "Recall@100": 0.90367,
#         "Recall@1000": 0.97833
#     },
#     "Precision": {
#         "P@1": 0.56667,
#         "P@3": 0.24556,
#         "P@5": 0.15733,
#         "P@10": 0.088,
#         "P@100": 0.01027,
#         "P@1000": 0.00111
#     },
#     "mrr": {
#         "MRR@1": 0.56667,
#         "MRR@3": 0.62667,
#         "MRR@5": 0.6355,
#         "MRR@10": 0.6447,
#         "MRR@100": 0.64971,
#         "MRR@1000": 0.65002
#     },
#     "hole": {
#         "Hole@1": 0.41,
#         "Hole@3": 0.71556,
#         "Hole@5": 0.80133,
#         "Hole@10": 0.85733,
#         "Hole@100": 0.9303,
#         "Hole@1000": 0.94051
#     }
# }
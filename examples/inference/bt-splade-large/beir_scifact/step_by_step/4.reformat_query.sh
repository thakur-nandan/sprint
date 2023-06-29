# Reformatting BEIR queries from Jsonl to TSV format
export SPRINT_HOME=../../../../../  # Path to the repo, please change it accordingly
export PYTHONPATH=$SPRINT_HOME:"${PYTHONPATH}"

export stage=reformat_query  # We need to transform the queries file from the BeIR format to the Pyserini format

export encoder_name='splade'
export encoder_ckpt_name='bt-splade-l'
export data_name='beir_scifact'  # beir data can be downloaded automatically

export split='test'  # We only need to reformat the test queries
export data_dir=datasets/beir/scifact  # The results will be saved under the same path: queries-test.tsv

export long_idenitifer=$data_name
export log_name=$stage.$long_idenitifer.log

nohup python -m sprint_toolkit.inference.$stage \
    --original_format 'beir' \
    --data_dir $data_dir \
    --topic_split $split \
    > $log_name &
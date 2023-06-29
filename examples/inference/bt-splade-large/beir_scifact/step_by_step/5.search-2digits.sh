# Step 5: Search the indexed collection using Pyserini with BEIR test queries
export SPRINT_HOME=../../../../../  # Path to the repo, please change it accordingly
export PYTHONPATH=$SPRINT_HOME:"${PYTHONPATH}"

export stage=search  # Adapted from the Pyserini README for reproducing uniCOIL on MSMARCO

export encoder_name='splade'
export encoder_ckpt_name='bt-splade-l'  # Here we use the query model, since the queries are required to be encoded
export data_name='beir_scifact'  # beir data can be downloaded automatically
export quantization=2digits  # The current encoding stage will output the original float weights without quantization

export ckpt_name=naver/efficient-splade-VI-BT-large-query # Naver's BT-SPLADE-L checkpoint for the query encoder
export long_idenitifer="$data_name-$encoder_ckpt_name-$quantization"
export log_name=$stage.$long_idenitifer.log
export index_dir=$long_idenitifer/index

export output_format=trec  # Could also be 'msmarco'. These formats are from Pyserini. 'trec' will keep also the scores
export output_path=$long_idenitifer/$stage/$output_format-format/run.tsv
export data_dir=datasets/beir/scifact
export queries_path=$data_dir/queries-test.tsv  # This is the input queries

nohup python -m sprint_toolkit.inference.$stage \
    --topics $queries_path \
    --encoder_name $encoder_name \
    --ckpt_name  $ckpt_name \
    --index $index_dir \
    --output $output_path \
    --impact \
    --hits 1000 --batch 36 --threads 12 \
    --output-format $output_format \
    > $log_name &
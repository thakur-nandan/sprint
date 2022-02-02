export SPARSE_RETRIEVAL_HOME=../../../../../  # Path to the repo, please change it accordingly
export PYTHONPATH=$SPARSE_RETRIEVAL_HOME:"${PYTHONPATH}"

export stage=search  # Adapted from the Pyserini README for reproducing uniCOIL on MSMARCO

export encoder_name='splade'
export encoder_ckpt_name='distilsplade_max'  # Here we use noexp model (i.e. no document expansion), since the documents are not expanded
export data_name='beir_nq'  # beir data can be downloaded automatically
export quantization=2digits  # The current encoding stage will output the original float weights without quantization

export ckpt_name=distilsplade_max
export long_idenitifer="$data_name-$encoder_ckpt_name-$quantization"
export log_name=$stage.$long_idenitifer.log
export index_dir=$long_idenitifer/index

export output_format=trec  # Could also be 'msmarco'. These formats are from Pyserini. 'trec' will keep also the scores
export output_path=$long_idenitifer/$stage/$output_format-format/run.tsv
export data_dir=datasets/beir/nq
export queries_path=$data_dir/queries-test.reformatted.tsv  # This is the input queries

nohup python -m sparse_retrieval.inference.$stage \
    --topics $queries_path \
    --encoder $ckpt_name \
    --index $index_dir \
    --output $output_path \
    --impact \
    --hits 1000 --batch 36 --threads 12 \
    --output-format $output_format \
    > $log_name &

# python -m inference.$stage \
#     --topics $queries_path \
#     --encoder $ckpt_name \
#     --index $index_dir \
#     --output $output_path \
#     --impact \
#     --hits 1000 --batch 36 --threads 12 \
#     --output-format $output_format
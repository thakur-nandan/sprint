export SPARSE_RETRIEVAL_HOME=../../../../  # Path to the repo, please change it accordingly
export PYTHONPATH=$SPARSE_RETRIEVAL_HOME:"${PYTHONPATH}"

export stage=reformat_query  # We need to transform the queries file from the BeIR format to the Pyserini format

export encoder_name='splade'
export encoder_ckpt_name='distilsplade_max'  # Here we use noexp model (i.e. no document expansion), since the documents are not expanded
export data_name='beir_scifact'  # beir data can be downloaded automatically

export data_dir=datasets/beir/scifact  # The results will be saved under the same path: queries-test.reformatted.tsv and queries-train.reformatted.tsv

export long_idenitifer=$data_name
export log_name=$stage.$long_idenitifer.log

nohup python -m inference.$stage \
    --original_format 'beir' \
    --data_dir $data_dir \
    > $log_name &
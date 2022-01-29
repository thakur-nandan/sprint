export SPARSE_RETRIEVAL_HOME=../../../../  # Path to the repo, please change it accordingly
export PYTHONPATH=$SPARSE_RETRIEVAL_HOME:"${PYTHONPATH}"

export stage=index  # Adapted from the Pyserini README for reproducing uniCOIL on MSMARCO

export encoder_name='unicoil'
export encoder_ckpt_name='unicoil_noexp'
export data_name='beir_scifact'
export quantization='b8'

export long_idenitifer=$data_name-$encoder_ckpt_name-$quantization

export collection_dir=$long_idenitifer/collection
export output_dir=$long_idenitifer/$stage

export log_name=$stage.$long_idenitifer.log

nohup python -m inference.$stage \
    -collection JsonVectorCollection \
    -input $collection_dir \
    -index $output_dir \
    -generator DefaultLuceneDocumentGenerator -impact -pretokenized \
    -threads 12 > $log_name &
# Step 3: Index the quantized collection using Pyserini
export SPRINT_HOME=../../../../../  # Path to the repo, please change it accordingly
export PYTHONPATH=$SPRINT_HOME:"${PYTHONPATH}"

export stage=index  # Adapted from the Pyserini README for reproducing uniCOIL on MSMARCO

export encoder_name='splade'
export encoder_ckpt_name='bt-splade-l'  # Here we use document model, since the documents are required to be encoded
export data_name='beir_scifact'  # beir data can be downloaded automatically
export quantization=2digits  # The current encoding stage will output the original float weights without quantization

export long_idenitifer=$data_name-$encoder_ckpt_name-$quantization

export collection_dir=$long_idenitifer/collection
export output_dir=$long_idenitifer/$stage

export log_name=$stage.$long_idenitifer.log

nohup python -m sprint_toolkit.inference.$stage \
    -collection JsonVectorCollection \
    -input $collection_dir \
    -index $output_dir \
    -generator DefaultLuceneDocumentGenerator -impact -pretokenized \
    -threads 12 > $log_name &
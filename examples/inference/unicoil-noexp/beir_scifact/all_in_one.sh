export SPARSE_RETRIEVAL_HOME=../../../../  # Path to the repo, please change it accordingly
export PYTHONPATH=$SPARSE_RETRIEVAL_HOME:"${PYTHONPATH}"

nohup python -m sparse_retrieval.inference.aio \
    --encoder_name unicoil \
    --ckpt_name castorini/unicoil-noexp-msmarco-passage \
    --data_name beir_scifact \
    --gpus 6 11 \
    --output_dir beir_scifact-unicoil_noexp \
    --do_quantization \
    --quantization_method range-nbits \
    --original_score_range 5 \
    --quantization_nbits 8 \
    --original_query_format beir \
    --topic_split test \
    > all_in_one.log &


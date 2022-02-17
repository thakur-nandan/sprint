if [ ! -d "distilsplade_max" ]; then
    wget https://download-de.europe.naverlabs.com/Splade_Release_Jan22/distilsplade_max.tar.gz
    tar -xvf distilsplade_max.tar.gz
fi

export SPARSE_RETRIEVAL_HOME=../../../../  # Path to the repo, please change it accordingly
export PYTHONPATH=$SPARSE_RETRIEVAL_HOME:"${PYTHONPATH}"

nohup python -m sparse_retrieval.inference.aio \
    --encoder_name splade \
    --ckpt_name distilsplade_max \
    --data_name beir_nq \
    --gpus 6 11 \
    --output_dir beir_nq-distilsplade_max \
    --do_quantization \
    --quantization_method ndigits-round \
    --ndigits 2 \
    --original_query_format beir \
    --topic_split test \
    > all_in_one.log &


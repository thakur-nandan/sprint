if [ ! -d "deepimpact-bert-base" ]; then
    wget https://public.ukp.informatik.tu-darmstadt.de/kwang/sparse-retrieval/checkpoints/deepimpact-bert-base.zip
    unzip deepimpact-bert-base.zip
fi

if [ ! -d "datasets/msmarco-deepimpact" ]; then
    mkdir -p datasets
    cd datasets
    wget https://public.ukp.informatik.tu-darmstadt.de/kwang/sparse-retrieval/data/msmarco-deepimpact.zip
    unzip msmarco-deepimpact.zip
    cd ..
fi

# Please make sure you have installed the repo

nohup python -m sparse_retrieval.inference.aio \
    --encoder_name deepimpact \
    --ckpt_name deepimpact-bert-base \
    --data_name beir \
    --data_dir datasets/msmarco-deepimpact \
    --gpus 0 \
    --output_dir msmarco-deepimpact-deepimpact \
    --do_quantization \
    --quantization_method ndigits-round \
    --ndigits 3 \
    --original_query_format beir \
    --topic_split dev \
    > all_in_one.log &

# TODO: change the quantization into range-nbits: https://github.com/DI4IR/SIGIR2021/blob/main/scripts/quantize.py
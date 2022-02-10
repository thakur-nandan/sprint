# Please make sure you have installed the repo

# Notice that it should be MS MARCO dev split
nohup python -m sparse_retrieval.inference.aio \
    --encoder_name tildev2 \
    --ckpt_name "ielab/TILDEv2-noExp" \
    --data_name beir_msmarco \
    --gpus 9 10 13 14 \
    --output_dir beir_msmarco-tildev2_noexp \
    --do_quantization \
    --quantization_method ndigits-round \
    --ndigits 2 \
    --original_query_format beir \
    --topic_split dev \
    > all_in_one.log &

# python -m sparse_retrieval.inference.aio \
#     --encoder_name tildev2 \
#     --ckpt_name "ielab/TILDEv2-noExp" \
#     --data_name beir_msmarco \
#     --gpus 9 \
#     --output_dir beir_msmarco-tildev2_noexp \
#     --do_quantization \
#     --quantization_method ndigits-round \
#     --ndigits 2 \
#     --original_query_format beir \
#     --topic_split dev


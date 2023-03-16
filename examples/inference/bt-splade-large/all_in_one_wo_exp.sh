# for dataset in nfcorpus
# do
#     python -m sprint.inference.aio \
#         --encoder_name splade \
#         --ckpt_name naver/efficient-splade-VI-BT-large-query naver/efficient-splade-VI-BT-large-doc \
#         --data_name beir_${dataset} \
#         --data_dir /store2/scratch/n3thakur/beir-datasets/${dataset} \
#         --gpus 1 \
#         --output_dir /store2/scratch/n3thakur/sparse-retrieval-results/efficient-splade-bt-large-wo-exp/${dataset} \
#         --do_quantization \
#         --quantization_method ndigits-round \
#         --ndigits 2 \
#         --original_query_format beir \
#         --topic_split test
# done

for dataset in msmarco
do
    for split in dev
    do
        # python -m sprint.inference.aio \
        #     --encoder_name splade \
        #     --ckpt_name /home/n3thakur/projects/splade/weights/distilsplade_max \
        #     --data_name beir_${dataset} \
        #     --data_dir /store2/scratch/n3thakur/beir-datasets/${dataset} \
        #     --gpus 4 6 \
        #     --output_dir /store2/scratch/n3thakur/sparse-retrieval-results/distilsplade_max/${dataset} \
        #     --do_quantization \
        #     --quantization_method ndigits-round \
        #     --ndigits 2 \
        #     --original_query_format beir \
        #     --topic_split ${split}

        python -m sprint.inference.aio \
            --encoder_name splade \
            --ckpt_name naver/efficient-splade-VI-BT-large-query naver/efficient-splade-VI-BT-large-doc \
            --data_name beir_${dataset} \
            --data_dir /store2/scratch/n3thakur/beir-datasets/${dataset} \
            --gpus 4 6 \
            --output_dir /store2/scratch/n3thakur/sparse-retrieval-results/efficient-splade-bt-large/${dataset} \
            --do_quantization \
            --quantization_method ndigits-round \
            --ndigits 2 \
            --original_query_format beir \
            --topic_split ${split}
    done
done
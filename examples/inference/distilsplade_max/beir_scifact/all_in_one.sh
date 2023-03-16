# if [ ! -d "distilsplade_max" ]; then
#     wget https://download-de.europe.naverlabs.com/Splade_Release_Jan22/distilsplade_max.tar.gz
#     tar -xvf distilsplade_max.tar.gz
# fi

# Please make sure you have installed the repo

python -m sprint.inference.aio \
    --encoder_name splade \
    --ckpt_name /home/n3thakur/projects/splade/weights/distilsplade_max \
    --data_name beir_arguana \
    --data_dir /store2/scratch/n3thakur/beir-datasets/arguana \
    --gpus 2 \
    --output_dir beir_arguana-distilsplade_max \
    --do_quantization \
    --quantization_method ndigits-round \
    --ndigits 2 \
    --original_query_format beir \
    --topic_split test


# {
#     "nDCG": {
#         "NDCG@1": 0.60333,
#         "NDCG@3": 0.65969,
#         "NDCG@5": 0.67204,
#         "NDCG@10": 0.6925,
#         "NDCG@100": 0.7202,
#         "NDCG@1000": 0.72753
#     },
#     "MAP": {
#         "MAP@1": 0.57217,
#         "MAP@3": 0.63391,
#         "MAP@5": 0.64403,
#         "MAP@10": 0.65444,
#         "MAP@100": 0.66071,
#         "MAP@1000": 0.66096
#     },
#     "Recall": {
#         "Recall@1": 0.57217,
#         "Recall@3": 0.70172,
#         "Recall@5": 0.73461,
#         "Recall@10": 0.79122,
#         "Recall@100": 0.92033,
#         "Recall@1000": 0.98
#     },
#     "Precision": {
#         "P@1": 0.60333,
#         "P@3": 0.25444,
#         "P@5": 0.16267,
#         "P@10": 0.08967,
#         "P@100": 0.01043,
#         "P@1000": 0.00111
#     },
#     "mrr": {
#         "MRR@1": 0.60333,
#         "MRR@3": 0.65722,
#         "MRR@5": 0.66306,
#         "MRR@10": 0.67052,
#         "MRR@100": 0.67503,
#         "MRR@1000": 0.67524
#     }
# }

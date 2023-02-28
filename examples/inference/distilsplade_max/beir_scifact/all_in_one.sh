if [ ! -d "distilsplade_max" ]; then
    wget https://download-de.europe.naverlabs.com/Splade_Release_Jan22/distilsplade_max.tar.gz
    tar -xvf distilsplade_max.tar.gz
fi

# Please make sure you have installed the repo

nohup python -m sparse_retrieval.inference.aio \
    --encoder_name splade \
    --ckpt_name distilsplade_max \
    --data_name beir_scifact \
    --gpus 0 \
    --output_dir beir_scifact-distilsplade_max \
    --do_quantization \
    --quantization_method ndigits-round \
    --ndigits 2 \
    --original_query_format beir \
    --topic_split test \
    > all_in_one.log &


# {
#     "nDCG": {
#         "NDCG@1": 0.60333,
#         "NDCG@2": 0.63895,
#         "NDCG@3": 0.65969,
#         "NDCG@5": 0.67204,
#         "NDCG@10": 0.6925,
#         "NDCG@20": 0.70403,
#         "NDCG@100": 0.7202,
#         "NDCG@1000": 0.72753
#     },
#     "MAP": {
#         "MAP@1": 0.57217,
#         "MAP@2": 0.61522,
#         "MAP@3": 0.63391,
#         "MAP@5": 0.64403,
#         "MAP@10": 0.65444,
#         "MAP@20": 0.65846,
#         "MAP@100": 0.66071,
#         "MAP@1000": 0.66096
#     },
#     "Recall": {
#         "Recall@1": 0.57217,
#         "Recall@2": 0.65078,
#         "Recall@3": 0.70172,
#         "Recall@5": 0.73461,
#         "Recall@10": 0.79122,
#         "Recall@20": 0.833,
#         "Recall@100": 0.92033,
#         "Recall@1000": 0.98
#     },
#     "Precision": {
#         "P@1": 0.60333,
#         "P@2": 0.35167,
#         "P@3": 0.25444,
#         "P@5": 0.16267,
#         "P@10": 0.08967,
#         "P@20": 0.0475,
#         "P@100": 0.01043,
#         "P@1000": 0.00111
#     },
#     "mrr": {
#         "MRR@1": 0.60333,
#         "MRR@2": 0.64167,
#         "MRR@3": 0.65722,
#         "MRR@5": 0.66306,
#         "MRR@10": 0.67052,
#         "MRR@20": 0.67304,
#         "MRR@100": 0.67503,
#         "MRR@1000": 0.67524
#     },
#     "latency": {
#         "latency_avg": 0.0632176259594659,
#         "query_word_length_avg": 13.85,
#         "binned": {
#             "word_length_bins": [
#                 5.0,
#                 7.6,
#                 10.2,
#                 12.8,
#                 15.4,
#                 18.0,
#                 20.6,
#                 23.2,
#                 25.8,
#                 28.400000000000002,
#                 31.0
#             ],
#             "freqs": [
#                 21,
#                 69,
#                 53,
#                 68,
#                 24,
#                 22,
#                 22,
#                 9,
#                 6,
#                 6
#             ],
#             "latencies_avg": [
#                 0.06157973294677536,
#                 0.062426611955935025,
#                 0.06298324124358223,
#                 0.0637137847172033,
#                 0.06537958721596762,
#                 0.06230686320131933,
#                 0.06198018672587427,
#                 0.06480219447926497,
#                 0.06900245506425784,
#                 0.06779847725027305
#             ],
#             "latencies_std": [
#                 0.007268265966041692,
#                 0.00695837791999461,
#                 0.007156900485436917,
#                 0.007058682842506954,
#                 0.00628885084604788,
#                 0.007282331879014841,
#                 0.007265829058465847,
#                 0.005582415388678566,
#                 0.0017274655407518776,
#                 3.0068899748332633e-05
#             ]
#         },
#         "batch_size": 61.06666666666667,
#         "processor": " Intel(R) Xeon(R) Platinum 8168 CPU @ 2.70GHz"
#     },
#     "index_size": "3.87MB"
# }
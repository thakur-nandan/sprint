if [ ! -d "distilsplade_max" ]; then
    wget https://download-de.europe.naverlabs.com/Splade_Release_Jan22/distilsplade_max.tar.gz
    tar -xvf distilsplade_max.tar.gz
fi

nohup python run.py > run.log &

# The evaluation results would be: (0.5 nDCG@10% higher than w/o removing stop words)
# {
#     "nDCG": {
#         "NDCG@1": 0.61333,
#         "NDCG@3": 0.66348,
#         "NDCG@5": 0.67439,
#         "NDCG@10": 0.69807,
#         "NDCG@100": 0.72434,
#         "NDCG@1000": 0.73203
#     },
#     "MAP": {
#         "MAP@1": 0.58217,
#         "MAP@3": 0.63957,
#         "MAP@5": 0.64897,
#         "MAP@10": 0.66102,
#         "MAP@100": 0.66644,
#         "MAP@1000": 0.66669
#     },
#     "Recall": {
#         "Recall@1": 0.58217,
#         "Recall@3": 0.70106,
#         "Recall@5": 0.72961,
#         "Recall@10": 0.795,
#         "Recall@100": 0.92033,
#         "Recall@1000": 0.98333
#     },
#     "Precision": {
#         "P@1": 0.61333,
#         "P@3": 0.25333,
#         "P@5": 0.16133,
#         "P@10": 0.09,
#         "P@100": 0.01043,
#         "P@1000": 0.00111
#     },
#     "mrr": {
#         "MRR@1": 0.61333,
#         "MRR@3": 0.66333,
#         "MRR@5": 0.66783,
#         "MRR@10": 0.67688,
#         "MRR@100": 0.68095,
#         "MRR@1000": 0.68117
#     }
# }
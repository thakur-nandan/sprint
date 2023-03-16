# Please make sure you have installed the repo
if [ ! -d "datasets/beir/msmarco" ]; then
    mkdir -p datasets/beir
    cd datasets/beir
    wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip
    unzip msmarco.zip
fi

if [ ! -f "datasets/tilde-trecdl2010-bm25_top1000/run.trec2019-bm25.res" ]; then
    mkdir -p datasets/tilde-trecdl2010-bm25_top1000
    cd datasets/tilde-trecdl2010-bm25_top1000
    wget https://raw.githubusercontent.com/ielab/TILDE/main/data/runs/run.trec2019-bm25.res
    cd ../..
fi

# Note that TREC-DL 2019 is also based on the collection of MS MARCO
# Note that we can use the test split from beir/msmarco
python -m sprint.inference.reformat_query \
    --original_format 'beir' \
    --data_dir datasets/beir/msmarco

python -m sprint.inference.rerank \
    --encoder_name tildev2 \
    --ckpt_name "ielab/TILDEv2-noExp" \
    --topics_path "datasets/beir/msmarco/queries-test.reformatted.tsv" \
    --topics_format anserini \
    --corpus_path "datasets/beir/msmarco/corpus.jsonl" \
    --output_dir "rerank/trec-format" \
    --device 0 \
    --retrieval_result_path "datasets/tilde-trecdl2010-bm25_top1000/run.trec2019-bm25.res"
    # --retrieval_result_path "datasets/tilde-trecdl2010-bm25_top1000/TILDEv2_rerank_BM25_top1000_dl2019.txt"
    # --retrieval_result_path "datasets/tilde-trecdl2010-bm25_top1000/bm25-top1000-dl2019-pass.txt"

python -m sprint.inference.evaluate \
    --result_path "rerank/trec-format/run.tsv" \
    --format trec \
    --qrels_path "datasets/beir/msmarco/qrels/test.tsv" \
    --output_dir "rerank/evaluation" \
    --k_values 1 3 5 10 100 1000

# This can give identical results to that run by the official code,
# for more details, please refer to https://github.com/ielab/TILDE/issues/1
# {
#     "nDCG": {
#         "NDCG@1": 0.5969,
#         "NDCG@3": 0.59461,
#         "NDCG@5": 0.6115,
#         "NDCG@10": 0.60579,
#         "NDCG@100": 0.56282,
#         "NDCG@1000": 0.63984
#     },
#     "MAP": {
#         "MAP@1": 0.01931,
#         "MAP@3": 0.05685,
#         "MAP@5": 0.08626,
#         "MAP@10": 0.13542,
#         "MAP@100": 0.34665,
#         "MAP@1000": 0.41857
#     },
#     "Recall": {
#         "Recall@1": 0.01931,
#         "Recall@3": 0.06061,
#         "Recall@5": 0.09705,
#         "Recall@10": 0.15333,
#         "Recall@100": 0.47498,
#         "Recall@1000": 0.73615
#     },
#     "Precision": {
#         "P@1": 0.81395,
#         "P@3": 0.76744,
#         "P@5": 0.76744,
#         "P@10": 0.71628,
#         "P@100": 0.34651,
#         "P@1000": 0.06516
#     },
#     "mrr": {
#         "MRR@1": 0.81395,
#         "MRR@3": 0.87597,
#         "MRR@5": 0.87597,
#         "MRR@10": 0.87597,
#         "MRR@100": 0.8782,
#         "MRR@1000": 0.8782
#     }
# }
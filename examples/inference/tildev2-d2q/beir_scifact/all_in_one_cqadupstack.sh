cudanum=7

for dataset in msmarco
do
    export output_results_path="/store2/scratch/n3thakur/sparse-retrieval-results/d2q-bm25-multifield/${dataset}"
    
    # python -m pyserini.search.lucene \
    #         --index ${output_results_path}/indexes/lucene-index-d2q-${dataset} \
    #         --topics /store2/scratch/n3thakur/beir-datasets/${dataset}/queries-dev.tsv \
    #         --output /store2/scratch/n3thakur/sparse-retrieval-results/tilde-d2q-v2-top-1000/${dataset}/d2q-top100/run.beir-v1.0.0-d2q-${dataset}-multifield.trec \
    #         --output-format trec \
    #         --batch 36 --threads 12 \
    #         --fields contents=1.0 title=1.0 \
    #         --remove-query --hits 101
    
    # python -m sprint.inference.reformat_query \
    # --original_format "beir" \
    # --data_dir "/store2/scratch/n3thakur/beir-datasets/${dataset}" \
    # --topic_split "dev"

    python -m sprint.inference.rerank \
        --encoder_name tildev2 \
        --ckpt_name "ielab/TILDEv2-docTquery-exp" \
        --topics_path "/store2/scratch/n3thakur/beir-datasets/${dataset}/queries-dev.tsv" \
        --topics_format anserini \
        --corpus_path "/store2/scratch/n3thakur/beir-datasets-d2q/${dataset}/corpus.jsonl" \
        --output_dir "/store2/scratch/n3thakur/sparse-retrieval-results/tilde-d2q-v2-top-1000/${dataset}/runs" \
        --batch_size 128 \
        --device ${cudanum} \
        --retrieval_result_path "/store2/scratch/n3thakur/sparse-retrieval-results/tilde-d2q-v2-top-1000/${dataset}/d2q-top100/run.beir-v1.0.0-d2q-${dataset}-multifield.trec"

    python -m sprint.inference.evaluate \
        --result_path "/store2/scratch/n3thakur/sparse-retrieval-results/tilde-d2q-v2-top-1000/${dataset}/runs/run.tsv" \
        --format trec \
        --qrels_path "/store2/scratch/n3thakur/beir-datasets/${dataset}/qrels/dev.tsv" \
        --output_dir "/store2/scratch/n3thakur/sparse-retrieval-results/tilde-d2q-v2-top-1000/${dataset}/eval" \
        --k_values 1 3 5 10 100 1000
done
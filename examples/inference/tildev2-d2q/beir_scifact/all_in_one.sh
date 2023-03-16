cudanum=2

for dataset in fiqa trec-covid robust04 dbpedia-entity
do
    export output_results_path="/store2/scratch/n3thakur/sparse-retrieval-results/d2q-bm25-multifield/${dataset}"
    
    # python -m pyserini.search.lucene \
    #         --index ${output_results_path}/indexes/lucene-index-d2q-${dataset} \
    #         --topics beir-v1.0.0-${dataset}-test \
    #         --output /store2/scratch/n3thakur/sparse-retrieval-results/tilde-d2q-v2-top-1000/${dataset}/d2q-top100/run.beir-v1.0.0-d2q-${dataset}-multifield.trec \
    #         --output-format trec \
    #         --batch 36 --threads 12 \
    #         --fields contents=1.0 title=1.0 \
    #         --remove-query --hits 101
    
    # python -m sprint.inference.reformat_query \
    # --original_format "beir" \
    # --data_dir "/store2/scratch/n3thakur/beir-datasets/${dataset}" \
    # --topic_split "test"

    python -m sprint.inference.rerank \
        --encoder_name tildev2 \
        --ckpt_name "ielab/TILDEv2-TILDE200-exp" \
        --topics_path "/store2/scratch/n3thakur/beir-datasets/${dataset}/queries-test.tsv" \
        --topics_format anserini \
        --corpus_path "/store2/scratch/n3thakur/beir-datasets-tilde/${dataset}/corpus.jsonl" \
        --output_dir "/store2/scratch/n3thakur/sparse-retrieval-results/tilde-tilde-v2-top-1000/${dataset}/runs" \
        --batch_size 32 \
        --device ${cudanum} \
        --retrieval_result_path "/store2/scratch/n3thakur/sparse-retrieval-results/tilde-bm25-multifield/${dataset}/runs/run.beir-v1.0.0-tilde-${dataset}-multifield.trec"

    python -m sprint.inference.evaluate \
        --result_path "/store2/scratch/n3thakur/sparse-retrieval-results/tilde-tilde-v2-top-1000/${dataset}/runs/run.tsv" \
        --format trec \
        --qrels_path "/store2/scratch/n3thakur/beir-datasets/${dataset}/qrels/test.tsv" \
        --output_dir "/store2/scratch/n3thakur/sparse-retrieval-results/tilde-tilde-v2-top-1000/${dataset}/eval" \
        --k_values 1 3 5 10 100 1000
done
# python -m pyserini.search.lucene \
#   --index beir-v1.0.0-bioasq-multifield \
#   --topics beir-v1.0.0-bioasq-test \
#   --output run.beir-multifield.bioasq.txt \
#   --output-format trec \
#   --batch 36 --threads 12 \
#   --hits 1000 --bm25 --remove-query --fields contents=1.0 title=1.0

python -m sprint.inference.reformat_query \
    --original_format "beir" \
    --data_dir "/store2/scratch/n3thakur/beir-datasets/bioasq" \
    --topic_split "test"

python -m sprint.inference.rerank \
    --encoder_name tildev2 \
    --ckpt_name "ielab/TILDEv2-noExp" \
    --topics_path "/store2/scratch/n3thakur/beir-datasets/bioasq/queries-test.tsv" \
    --topics_format anserini \
    --corpus_path "/store2/scratch/n3thakur/beir-datasets/bioasq/corpus.jsonl" \
    --output_dir "trec-format" \
    --batch_size 128 \
    --device 7 \
    --retrieval_result_path "run.beir-multifield.bioasq.txt"

python -m sprint.inference.evaluate \
    --result_path "trec-format/run.tsv" \
    --format trec \
    --qrels_path "/store2/scratch/n3thakur/beir-datasets/bioasq/qrels/test.tsv" \
    --output_dir "rerank/bioasq/evaluation" \
    --k_values 1 3 5 10 100 1000
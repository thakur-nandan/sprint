for dataset in signal1m nq dbpedia-entity fever hotpotqa
do
    export data_path="/store2/scratch/n3thakur/beir-datasets/${dataset}"
    export d2q_filepath="/store2/scratch/n3thakur/beir-datasets/${dataset}/top20-gen-queries.jsonl"
    export model_name_or_path="/home/n3thakur/projects/sparse-retrieval/scripts/deepimpact-d2q/deepimpact-bert-base"
    export output_path="/store2/scratch/n3thakur/beir-datasets-deepimpact/${dataset}"
    export output_results_path="/store2/scratch/n3thakur/sparse-retrieval-results/deepimpact-d2q/${dataset}"
    export split="test"
    
    python preprocessing.py \
        --dataset_dir ${data_path} \
        --output_path ${output_path} \
        --d2q_filepath ${d2q_filepath}
    
    python -m sparse_retrieval.inference.aio \
        --encoder_name deepimpact \
        --ckpt_name ${model_name_or_path} \
        --data_name beir/${dataset} \
        --gpus 0 \
        --train_data_dir ${output_path} \
        --eval_data_dir ${data_path} \
        --output_dir ${output_results_path} \
        --do_quantization \
        --quantization_method range-nbits \
        --original_score_range -1 \
        --quantization_nbits 8 \
        --original_query_format beir \
        --topic_split ${split}
done
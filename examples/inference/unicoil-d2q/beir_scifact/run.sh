for dataset in msmarco
do
    export data_path="/store2/scratch/n3thakur/beir-datasets/${dataset}"
    export d2q_filepath="/store2/scratch/n3thakur/beir-datasets/${dataset}/top40-gen-queries.jsonl"
    export model_name_or_path="castorini/unicoil-msmarco-passage"
    export output_path="/store2/scratch/n3thakur/beir-datasets-unicoil/${dataset}"
    export output_results_path="/store2/scratch/n3thakur/sparse-retrieval-results/unicoil-d2q/${dataset}"
    export split="test"
    
    # python preprocessing.py \
    #     --dataset_dir ${data_path} \
    #     --model_name_or_path ${model_name_or_path} \
    #     --output_path ${output_path} \
    #     --d2q_filepath ${d2q_filepath}
    
    python run.py \
        --dataset ${dataset} \
        --split ${split} \
        --model_name_or_path ${model_name_or_path} \
        --train_data_dir ${output_path} \
        --eval_data_dir ${data_path} \
        --output_results_path ${output_results_path} \
        --gpus 4 5 6
done
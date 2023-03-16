for dataset in dbpedia-entity
do
    export data_path="/store2/scratch/n3thakur/beir-datasets/${dataset}"
    export tilde_filepath="/store2/scratch/n3thakur/beir-datasets/${dataset}/collection-tilde-expanded-top200.jsonl"
    export model_name_or_path="ielab/unicoil-tilde200-msmarco-passage"
    export output_path="/store2/scratch/n3thakur/beir-datasets-unicoil-tilde200/${dataset}"
    export output_results_path="/store2/scratch/n3thakur/sparse-retrieval-results/unicoil-tilde200/${dataset}"
    export split="test"
    
    python preprocessing_tilde.py \
        --dataset_dir ${data_path} \
        --model_name_or_path ${model_name_or_path} \
        --output_path ${output_path} \
        --tilde_filepath ${tilde_filepath}
    
    python run.py \
        --dataset ${dataset} \
        --split ${split} \
        --model_name_or_path ${model_name_or_path} \
        --train_data_dir ${output_path} \
        --eval_data_dir ${data_path} \
        --output_results_path ${output_results_path} \
        --gpus 4 6
done
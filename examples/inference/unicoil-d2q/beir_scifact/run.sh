for dataset in nfcorpus trec-covid fiqa scidocs arguana trec-news robust04
do
    export data_path="/home/ukp/thakur/projects/sbert_retriever/datasets-expanded/${dataset}"
    export model_name_or_path="castorini/unicoil-msmarco-passage"
    export out_path="/home/ukp/thakur/projects/sparse-retrieval/examples/inference/unicoil-d2q/beir_scifact/preprocessed"
    export split="test"
    
    python preprocessing.py ${dataset} $data_path $split $model_name_or_path $out_path
    python run.py ${dataset} $out_path $split $model_name_or_path
done
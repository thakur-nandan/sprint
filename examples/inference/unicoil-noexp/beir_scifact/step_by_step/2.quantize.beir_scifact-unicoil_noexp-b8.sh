export SPARSE_RETRIEVAL_HOME=../../../../  # Path to the repo, please change it accordingly
export PYTHONPATH=$SPARSE_RETRIEVAL_HOME:"${PYTHONPATH}"

export stage=quantize  # Actually it is also OK to escape quantization (i.e. quantize the document term weights into integers)

export encoder_name='unicoil'
export encoder_ckpt_name='unicoil_noexp'
export data_name='beir_scifact'
export quantization_from='float'
export quantization_to='b8'  # Now the quantization stage will quantize the term weights into integers

# These two are the parameters of the quantization. Refer to the code for more details:
export quantization_method='range-nbits'
export original_score_range=5  # The original term weights should not be larger than this
export quantization_nbits=8  # How many bits used to do quantization

export long_idenitifer_from=$data_name-$encoder_ckpt_name-$quantization_from
export long_idenitifer_to=$data_name-$encoder_ckpt_name-$quantization_to

export collection_dir=$long_idenitifer_from/collection
export output_dir=$long_idenitifer_to/collection

export log_name=$stage.$long_idenitifer_to.log

nohup python -m inference.$stage \
    --collection_dir $collection_dir \
    --output_dir $output_dir \
    --method $quantization_method \
    --original_score_range $original_score_range \
    --quantization_nbits $quantization_nbits \
    --nprocs 12 > $log_name &
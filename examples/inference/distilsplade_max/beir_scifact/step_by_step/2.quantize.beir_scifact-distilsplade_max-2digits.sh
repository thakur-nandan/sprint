export SPRINT_HOME=../../../../../  # Path to the repo, please change it accordingly
export PYTHONPATH=$SPRINT_HOME:"${PYTHONPATH}"

export stage=quantize  # Actually it is also OK to escape quantization (i.e. quantize the document term weights into integers)

export encoder_name='splade'
export encoder_ckpt_name='distilsplade_max'  # Here we use noexp model (i.e. no document expansion), since the documents are not expanded
export data_name='beir_scifact'  # beir data can be downloaded automatically
export quantization_from='float'
export quantization_to='2digits'  # Now the quantization stage will quantize the term weights into integers

# These two are the parameters of the quantization. Refer to the code for more details:
export quantization_method='ndigits-round'
export ndigits=2

export long_idenitifer_from=$data_name-$encoder_ckpt_name-$quantization_from
export long_idenitifer_to=$data_name-$encoder_ckpt_name-$quantization_to

export collection_dir=$long_idenitifer_from/collection
export output_dir=$long_idenitifer_to/collection

export log_name=$stage.$long_idenitifer_to.log

nohup python -m sprint.inference.$stage \
    --collection_dir $collection_dir \
    --output_dir $output_dir \
    --method $quantization_method \
    --ndigits $ndigits \
    --nprocs 12 > $log_name &
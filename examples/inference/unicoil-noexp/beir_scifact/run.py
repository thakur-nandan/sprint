from sparse_retrieval.inference import aio


if __name__ == '__main__':  # aio.run can only be called within __main__
    aio.run(
        encoder_name='unicoil',
        ckpt_name=['castorini/unicoil-noexp-msmarco-passage',],
        data_name='beir/scifact',
        gpus=[0, 1],
        output_dir='beir_scifact-unicoil_noexp',
        do_quantization=True,
        quantization_method='range-nbits',  # So the doc term weights will be quantized by `(term_weights / 5) * (2 ** 8)`
        original_score_range=5,
        quantization_nbits=8,
        original_query_format='beir',
        topic_split='test'
    )
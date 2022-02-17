from sparse_retrieval.inference import aio
import sys

if __name__ == '__main__':  # aio.run can only be called within __main__
    aio.run(
        encoder_name='unicoil',
        ckpt_name=sys.argv[4],
        data_name='beir/{}'.format(sys.argv[1]),
        data_dir=sys.argv[2],
        gpus=[0,1],
        output_dir='/home/ukp/thakur/projects/sparse-retrieval/results/zero-shot/unicoil-d2q-exp/{}'.format(sys.argv[1]),
        do_quantization=True,
        quantization_method='range-nbits',  # So the doc term weights will be quantized by `(term_weights / 5) * (2 ** 8)`
        original_score_range=5,
        quantization_nbits=8,
        original_query_format='beir',
        topic_split=sys.argv[3]
    )
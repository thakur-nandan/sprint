from sparse_retrieval.inference import aio
from sparse_retrieval.inference.encoder_builders import register
from splade_stopwords import splade_stopwords

register('splade_stopwords', splade_stopwords)  # Register your custom encoders so that the repo knows it

if __name__ == '__main__':
    aio.run(
        encoder_name='splade_stopwords',
        ckpt_name='distilsplade_max',
        data_name='beir/scifact',
        gpus=[5, 6],
        output_dir='beir_scifact-distilsplade_max-stopwords',
        do_quantization=True,
        quantization_method='ndigits-round',
        ndigits=2,
        original_query_format='beir',
        topic_split='test'
    )
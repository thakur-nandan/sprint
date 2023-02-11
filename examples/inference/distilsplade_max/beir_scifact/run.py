from genericpath import exists
import os
from sparse_retrieval.inference import aio
from beir.util import download_url

def download_checkpoint():
    ckpt_name = 'distilsplade_max'
    if os.path.exists(ckpt_name):
        return
    
    os.makedirs(ckpt_name, exist_ok=True)
    os.system(f'cd {ckpt_name}')
    
    to_download = [
        'https://github.com/naver/splade/raw/main/weights/distilsplade_max/pytorch_model.bin',
        'https://github.com/naver/splade/raw/main/weights/distilsplade_max/config.json',
        'https://github.com/naver/splade/raw/main/weights/distilsplade_max/special_tokens_map.json',
        'https://github.com/naver/splade/raw/main/weights/distilsplade_max/tokenizer_config.json',
        'https://github.com/naver/splade/raw/main/weights/distilsplade_max/vocab.txt'
    ]
    for url in to_download:
        os.system(f'wget {url} -P {ckpt_name}')
    os.system('cd ..')


if __name__ == '__main__':  # aio.run can only be called within __main__
    download_checkpoint()

    aio.run(
        encoder_name='splade',
        ckpt_name=['distilsplade_max',],
        data_name='beir/scifact',
        gpus=[11, 12],
        output_dir='beir_scifact-distilsplade_max',
        do_quantization=True,
        quantization_method='ndigits-round',
        ndigits=2,
        original_query_format='beir',
        topic_split='test'
    )
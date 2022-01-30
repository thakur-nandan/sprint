from pyserini.encode import UniCoilQueryEncoder, UniCoilDocumentEncoder, SpladeQueryEncoder
from .methods import SpladeDocumentEncoder
from functools import partial


def unicoil(ckpt_name, gpu):
    document_encoder = UniCoilDocumentEncoder(ckpt_name, device=gpu)

    def encoder_fn(texts):
        return document_encoder.encode(texts)

    return encoder_fn

def splade(ckpt_name, gpu):
    print('WARNING: Since the online checkpoints are only avaible at https://github.com/naver/splade/raw/main/weights/splade_max,'
        'please download them and use a local path for `ckpt_name`')
    document_encoder = SpladeDocumentEncoder(ckpt_name, device=gpu)

    def encoder_fn(texts):
        return document_encoder.encode(texts)
    
    return encoder_fn


def build(encoder_name, ckpt_name):
    if encoder_name == 'unicoil':
        return partial(unicoil, ckpt_name)
    elif encoder_name == 'splade':
        return partial(splade, ckpt_name)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    encode_fn_builder = build('unicoil', 'castorini/unicoil-noexp-msmarco-passage')
    encode_fn = encode_fn_builder(0)
    text = "Minority interest The reporting of 'minority interest' is a consequence of the requirement by accounting standards to 'fully' consolidate partly owned subsidiaries. Full consolidation, as opposed to partial consolidation, results in financial statements that are constructed as if the parent corporation fully owns these partly owned subsidiaries; except for two line items that reflect partial ownership of subsidiaries: net income to common shareholders and common equity. The two minority interest line items are the net difference between what would have been the common equity and net income to common, if all subsidiaries were fully owned, and the actual ownership of the group. All the other line items in the financial statements assume a fictitious 100% ownership."
    print(encode_fn(text))
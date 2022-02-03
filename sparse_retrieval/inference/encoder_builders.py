from json import encoder
from pydoc import Doc
from numpy import uint
from pyserini.encode import UniCoilQueryEncoder, UniCoilDocumentEncoder, SpladeQueryEncoder, QueryEncoder, DocumentEncoder
from typing import Union
from .methods import SpladeDocumentEncoder
from functools import partial


def unicoil(ckpt_name, etype, device='cpu'):
    if etype == 'query':
        return UniCoilQueryEncoder(ckpt_name, device=device)        
    elif etype == 'document':
        return UniCoilDocumentEncoder(ckpt_name, device=device)
    else:
        raise ValueError

def splade(ckpt_name, etype, device='cpu'):
    print('WARNING: Since the online checkpoints are only avaible at https://github.com/naver/splade/raw/main/weights/splade_max,'
        'please download them and use a local path for `ckpt_name`')
    if etype == 'query':
        return SpladeQueryEncoder(ckpt_name, device=device)        
    elif etype == 'document':
        return SpladeDocumentEncoder(ckpt_name, device=device)
    else:
        raise ValueError

ENCODER_MAPPING = {
    'unicoil': unicoil,
    'splade': splade
}

def get_builder(encoder_name, ckpt_name, etype) -> Union[QueryEncoder, DocumentEncoder]:
    assert etype in ['query', 'document']
    
    encoder_name = encoder_name.lower()
    assert encoder_name in ENCODER_MAPPING

    return partial(ENCODER_MAPPING[encoder_name], ckpt_name, etype)


if __name__ == '__main__':
    encoder_builder = get_builder('unicoil', 'castorini/unicoil-noexp-msmarco-passage', 'document')
    encoder: DocumentEncoder = encoder_builder(0)
    text = "Minority interest The reporting of 'minority interest' is a consequence of the requirement by accounting standards to 'fully' consolidate partly owned subsidiaries. Full consolidation, as opposed to partial consolidation, results in financial statements that are constructed as if the parent corporation fully owns these partly owned subsidiaries; except for two line items that reflect partial ownership of subsidiaries: net income to common shareholders and common equity. The two minority interest line items are the net difference between what would have been the common equity and net income to common, if all subsidiaries were fully owned, and the actual ownership of the group. All the other line items in the financial statements assume a fictitious 100% ownership."
    print(encoder.encode([text]))
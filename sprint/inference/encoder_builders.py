from json import encoder
from pydoc import Doc
from telnetlib import DO
from numpy import uint
from pyserini.encode import QueryEncoder, DocumentEncoder
from typing import Callable, Union
from .methods import (
    SpladeDocumentEncoder, 
    SpladeQueryEncoder,
    SPARTAQueryEncoder, 
    SPARTADocumentEncoder, 
    TILDEv2QueryEncoder, 
    TILDEv2DocumentEncoder, 
    DeepImpactQueryEncoder, 
    DeepImpactDocumentEncoder,
    UniCoilQueryEncoder,
    UniCoilDocumentEncoder
)
from functools import partial


def unicoil(ckpt_name, etype, device='cpu') -> Union[QueryEncoder, DocumentEncoder]:
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

def sparta(ckpt_name, etype, device='cpu'):
    if etype == 'query':
        return SPARTAQueryEncoder(ckpt_name, device=device)        
    elif etype == 'document':
        return SPARTADocumentEncoder(ckpt_name, device=device)
    else:
        raise ValueError

def tildev2(ckpt_name, etype, device='cpu'):
    if etype == 'query':
        return TILDEv2QueryEncoder(ckpt_name, device=device)        
    elif etype == 'document':
        return TILDEv2DocumentEncoder(ckpt_name, device=device)
    else:
        raise ValueError

def deepimpact(ckpt_name, etype, device='cpu'):
    if etype == 'query':
        return DeepImpactQueryEncoder(ckpt_name, device=device)        
    elif etype == 'document':
        return DeepImpactDocumentEncoder(ckpt_name, device=device)
    else:
        raise ValueError


ENCODER_MAPPING = {
    'unicoil': unicoil,
    'splade': splade,
    'sparta': sparta,
    'tildev2': tildev2,
    'deepimpact': deepimpact
}

def register(encoder_name, builder_fn: Callable[[str, str, Union[str, int]], Union[QueryEncoder, DocumentEncoder]]):
    assert encoder_name not in ENCODER_MAPPING
    ENCODER_MAPPING[encoder_name] = builder_fn

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
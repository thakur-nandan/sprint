from typing import Dict, List
from nltk.corpus import stopwords
from collections import defaultdict
from sparse_retrieval.inference.methods import SpladeQueryEncoder, SpladeDocumentEncoder

stopwords = set(stopwords.words('english'))

def remove_stopwords(term_weights: Dict[str, float]):
    for term in list(term_weights.keys()):
        if term in stopwords:
            term_weights.pop(term)


class SpladeStopWordsQueryEncoder(SpladeQueryEncoder):

    def encode(self, text, **kwargs) -> Dict[str, float]:
        term_weights: Dict[str, float] = super().encode(text, **kwargs)
        remove_stopwords(term_weights)
        return term_weights


class SpladeStopWordsDocumentEncoder(SpladeDocumentEncoder):

    def encode(self, texts, **kwargs) -> List[Dict[str, float]]:
        term_weights_batch = super().encode(texts, **kwargs)
        map(remove_stopwords, term_weights_batch)
        return term_weights_batch


def splade_stopwords(ckpt_name, etype, device='cpu'):
    if etype == 'query':
        return SpladeStopWordsQueryEncoder(ckpt_name, device=device)        
    elif etype == 'document':
        return SpladeStopWordsDocumentEncoder(ckpt_name, device=device)
    else:
        raise ValueError
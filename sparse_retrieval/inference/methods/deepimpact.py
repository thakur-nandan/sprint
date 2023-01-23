# Adapted from https://github.com/DI4IR/SIGIR2021/blob/main/src/model.py

import os
from typing import Optional, Union
import torch
import torch.nn as nn

from random import sample, shuffle, randint

from itertools import accumulate
from transformers import AutoConfig, AutoTokenizer, AutoModel, PreTrainedModel
from pyserini.encode import QueryEncoder, DocumentEncoder
import re

MAX_LENGTH = 300

STOPLIST = ["a", "about", "also", "am", "an", "and", "another", "any", "anyone", "are", "aren't", "as", "at", "be",
            "been", "being", "but", "by", "despite", "did", "didn't", "do", "does", "doesn't", "doing", "done", "don't",
            "each", "etc", "every", "everyone", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
            "haven't", "having", "he", "he'd", "he'll", "her", "here", "here's", "hers", "herself", "he's",
            "him", "himself", "his", "however", "i", "i'd", "if", "i'll", "i'm", "in", "into", "is", "isn't", "it",
            "its", "it's", "itself", "i've", "just", "let's", "like", "lot", "may", "me", "might", "mightn't",
            "my", "myself", "no", "nor", "not", "of", "on", "onto", "or", "other", "ought", "oughtn't", "our", "ours",
            "ourselves", "out", "over", "shall", "shan't", "she", "she'd", "she'll", "she's", "since", "so", "some",
            "something", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
            "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through",
            "tht", "to", "too", "usually", "very", "via", "was", "wasn't", "we", "we'd", "well", "we'll", "were",
            "we're", "weren't", "we've", "will", "with", "without", "won't", "would", "wouldn't", "yes", "yet", "you",
            "you'd", "you'll", "your", "you're", "yours", "yourself", "yourselves", "you've"]

printable = set('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ')
printableX = set('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-. ')
printable3X = set('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,.- ')

printableD = set('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.- ')
printable3D = set('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.- ')

STOPLIST_ = list(map(lambda s: ''.join(filter(lambda x: x in printable, s)), STOPLIST))

STOPLIST = {}
for w in STOPLIST_:
    STOPLIST[w] = True

def cleanD(s, join=True):
    s = [(x.lower() if x in printable3X else ' ') for x in s]
    s = [(x if x in printableX else ' ' + x + ' ') for x in s]
    s = ''.join(s).split()
    s = [(w if '.' not in w else (' . ' if len(max(w.split('.'), key=len)) > 1 else '').join(w.split('.'))) for w in s]
    s = ' '.join(s).split()
    s = [(w if '-' not in w else w.replace('-', '') + ' ( ' + ' '.join(w.split('-')) + ' ) ') for w in s]
    s = ' '.join(s).split()
    # s = [w for w in s if w not in STOPLIST]

    return ' '.join(s) if join else s


def cleanQ(s, join=True):
    s = [(x.lower() if x in printable3D else ' ') for x in s]
    s = [(x if x in printableD else ' ' + x + ' ') for x in s]
    s = ''.join(s).split()
    s = [(w if '.' not in w else (' ' if len(max(w.split('.'), key=len)) > 1 else '').join(w.split('.'))) for w in s]
    s = ' '.join(s).split()
    s = [(w if '-' not in w else (' ' if len(min(w.split('-'), key=len)) > 1 else '').join(w.split('-'))) for w in s]
    s = ' '.join(s).split()
    s = [w for w in s if w not in STOPLIST]

    return ' '.join(s) if join else s


class DeepImpact(PreTrainedModel):
    config_class = AutoConfig
    # base_model_prefix = 'coil_encoder'
    base_model_prefix = 'deep_impact'

    def __init__(self, config, *model_args, **model_kwargs):
        super(DeepImpact, self).__init__(config)
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # self.bert = BertModel(config)
        # self.tokenizer = AutoTokenizer.from_config(config)
        self.bert = AutoModel.from_config(config)
        self.pre_classifier = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.impact_score_encoder = nn.Sequential(
        #     nn.Linear(config.hidden_size, config.hidden_size),
        #     nn.ReLU(),
        #     nn.Dropout(config.hidden_dropout_prob),
        #     nn.Linear(config.hidden_size, 1),
        #     nn.ReLU()
        # )
        self.init_weights()
                
    def impact_score_encoder(self, hs):
        for module in [self.pre_classifier, self.relu, self.dropout, self.classifier, self.relu]:
            hs = module(hs)
        return hs

    def convert_example(self, d, max_seq_length):
        max_length = min(MAX_LENGTH, max_seq_length)
        inputs = self.tokenizer.encode_plus(d, add_special_tokens=True, max_length=max_length, truncation=True)

        padding_length = max_length - len(inputs["input_ids"])
        attention_mask = ([1] * len(inputs["input_ids"])) + ([0] * padding_length)
        input_ids = inputs["input_ids"] + ([0] * padding_length)
        token_type_ids = inputs["token_type_ids"] + ([0] * padding_length)

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}

    def tokenize(self, q, d):
        query_tokens = list(set(cleanQ(q).strip().split()))  # [:10]

        content = cleanD(d).strip()
        doc_tokens = content.split()

        # NOTE: The following line accounts for CLS!
        tokenized = self.tokenizer.tokenize(content)
        word_indexes = list(accumulate([-1] + tokenized, lambda a, b: a + int(not b.startswith('##'))))
        match_indexes = list(set([doc_tokens.index(t) for t in query_tokens if t in doc_tokens]))
        term_indexes = [word_indexes.index(idx) for idx in match_indexes]

        a = [idx for i, idx in enumerate(match_indexes) if term_indexes[i] < MAX_LENGTH]
        b = [idx for idx in term_indexes if idx < MAX_LENGTH]

        return content, tokenized, a, b, len(word_indexes) + 2

    def tokenize_documents(self, d):
        d = cleanD(d, join=False)
        content = ' '.join(d)
        tokenized_content = self.tokenizer.tokenize(content)

        terms = list(set([(t, d.index(t)) for t in d]))  # Quadratic!
        word_indexes = list(accumulate([-1] + tokenized_content, lambda a, b: a + int(not b.startswith('##'))))
        terms = [(t, word_indexes.index(idx)) for t, idx in terms]
        terms = [(t, idx) for (t, idx) in terms if idx < MAX_LENGTH]

        return tokenized_content, terms

    def forward(self, Q, D):
        bsize = len(Q)
        pairs = []
        X, pfx_sum, pfx_sumX = [], [], []
        total_size, total_sizeX, max_seq_length = 0, 0, 0

        doc_partials = []
        pre_pairs = []

        for q, d in zip(Q, D):
            tokens, tokenized, term_idxs, token_idxs, seq_length = self.tokenize(q, d)
            max_seq_length = max(max_seq_length, seq_length)

            pfx_sumX.append(total_sizeX)
            total_sizeX += len(term_idxs)

            tokens_split = tokens.split()

            doc_partials.append([(total_size + idx, tokens_split[i]) for idx, i in enumerate(term_idxs)])
            total_size += len(doc_partials[-1])
            pfx_sum.append(total_size)

            pre_pairs.append((tokenized, token_idxs))

        for tokenized, token_idxs in pre_pairs:
            pairs.append(self.convert_example(tokenized, max_seq_length))
            X.append(token_idxs)

        input_ids = torch.tensor([f['input_ids'] for f in pairs], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor([f['attention_mask'] for f in pairs], dtype=torch.long).to(self.device)
        token_type_ids = torch.tensor([f['token_type_ids'] for f in pairs], dtype=torch.long).to(self.device)

        outputs = self.bert.forward(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        hidden_state = outputs[0]

        def one(i):
            if len(X[i]) > 0:
                l = [hidden_state[i, j] for j in X[i]]  # + [mismatch_scores[i, j] for j in all_mismatches[i]]
                return torch.stack(l)
            return torch.tensor([]).to(self.device)

        pooled_output = torch.cat([one(i) for i in range(bsize)])

        bsize = len(pooled_output)

        if bsize == 0:
            term_scores = []
            for doc in doc_partials:
                term_scores.append([])
                for (idx, term) in doc:
                    term_scores[-1].append((term, 0.0))

            return torch.tensor([[0.0]] * len(Q)).to(self.device), term_scores

        y_score = self.impact_score_encoder(pooled_output)

        x = torch.arange(bsize).expand(len(pfx_sum), bsize) < torch.tensor(pfx_sum).unsqueeze(1)
        y = torch.arange(bsize).expand(len(pfx_sum), bsize) >= torch.tensor([0] + pfx_sum[:-1]).unsqueeze(1)
        mask = (x & y).to(self.device)

        y_scorex = list(y_score.cpu())
        term_scores = []
        for doc in doc_partials:
            term_scores.append([])
            for (idx, term) in doc:
                term_scores[-1].append((term, y_scorex[idx]))

        return (mask.type(torch.float32) @ y_score), term_scores #, ordered_terms #, num_exceeding_fifth

    def index(self, D, max_seq_length):
        bsize = len(D)
        offset = 0
        pairs, X = [], []

        for tokenized_content, terms in D:
            terms = [(t, idx, offset + pos) for pos, (t, idx) in enumerate(terms)]
            offset += len(terms)
            pairs.append(self.convert_example(tokenized_content, max_seq_length))
            X.append(terms)

        input_ids = torch.tensor([f['input_ids'] for f in pairs], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor([f['attention_mask'] for f in pairs], dtype=torch.long).to(self.device)
        token_type_ids = torch.tensor([f['token_type_ids'] for f in pairs], dtype=torch.long).to(self.device)

        outputs = self.bert.forward(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        hidden_state = outputs[0]
        pooled_output = torch.cat([hidden_state[i, list(map(lambda x: x[1], X[i]))] for i in range(bsize)])

        y_score = self.impact_score_encoder(pooled_output)
        y_score = y_score.squeeze().cpu().numpy().tolist()
        # term_scores = [[(term, y_score[pos]) for term, _, pos in terms] for terms in X]
        # term_scores = [{term: round(y_score[pos], 3) for term, _, pos in terms} for terms in X]
        term_scores = [{term: y_score[pos] for term, _, pos in terms} for terms in X]
        return term_scores


class DeepImpactDocumentEncoder(DocumentEncoder):

    def __init__(self, model_name, device) -> None:
        self.model: DeepImpact = DeepImpact.from_pretrained(model_name)
        self.device = device
        self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model.tokenizer = self.tokenizer
        self.reverse_voc = {v: k for k, v in self.tokenizer.vocab.items()}

    def encode(self, texts, **kwargs):
        super_batch = list(map(self.model.tokenize_documents, texts))
        max_seq_length = max([len(tokenized_content) for tokenized_content, terms in super_batch]) + 2
        return self.model.index(super_batch, max_seq_length)

class DeepImpactQueryEncoder(QueryEncoder):

    def __init__(self, model_name_or_path, device='cpu'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)

    def encode(self, text, **kwargs):
        query_tokens = list(set(cleanQ(text).strip().split()))  # [:10]
        return dict(zip(query_tokens, [1] * len(query_tokens)))


if __name__ == '__main__':
    query_encoder = DeepImpactQueryEncoder('deepimpact-bert-base')
    doc_encoder = DeepImpactDocumentEncoder('deepimpact-bert-base', 'cuda')

    q = 'What is the presence of communication communication communication amid scientific minds?'
    d = 'The presence of communication amid scientific minds was equally important to the success of the Manhattan Project as scientific intellect was. The only cloud hanging over the impressive achievement of the atomic researchers and engineers is what their success truly meant; hundreds of thousands of innocent lives obliterated. [SEP] how be world are results public an have which it want result significant in were they communicate on amongst involved affect believe power impact successful importance scientists bomb contribute outcome why know effect so who effects about did for purpose valuable following most accomplishment a communications'

    with torch.no_grad():
        tw_q = query_encoder.encode(q)
        tw_d = doc_encoder.encode([d])[0]
    
    print(tw_q)
    print(tw_d)
    tw_shared = {t: tw_d[t] for t in tw_q if t in tw_d}
    print(tw_shared)
    print(sum(tw_shared.values()))

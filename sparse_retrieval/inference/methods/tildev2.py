# Adapted from https://github.com/ielab/TILDE/blob/main/modelingv2.py

import numpy as np
import torch
from transformers import AutoTokenizer, PreTrainedModel, BertConfig, BertModel, PreTrainedTokenizer
from transformers.trainer import Trainer
from torch.utils.data import DataLoader
from typing import Optional
from pyserini.encode import QueryEncoder, DocumentEncoder
import re


def get_stop_ids(tok: PreTrainedTokenizer):
    # hard code for now, from nltk.corpus import stopwords, stopwords.words('english')
    stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
                      "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
                      'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
                      'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
                      'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
                      'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
                      'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
                      'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
                      'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
                      'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
                      'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
                      "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren',
                      "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't",
                      'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
                      "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't",
                      'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])
    # keep some common words in ms marco questions
    stop_words.difference_update(["where", "how", "what", "when", "which", "why", "who"])

    vocab = tok.get_vocab()
    tokens = vocab.keys()

    stop_ids = []

    for stop_word in stop_words:
        ids = tok(stop_word, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            stop_ids.append(ids[0])

    for token in tokens:
        token_id = vocab[token]
        if token_id in stop_ids:
            continue
        if token == '##s':  # remove 's' suffix
            stop_ids.append(token_id)
        if token[0] == '#' and len(token) > 1:  # skip most of subtokens
            continue
        if not re.match("^[A-Za-z0-9_-]*$", token):  # remove numbers, symbols, etc..
            stop_ids.append(token_id)

    return set(stop_ids)


class TILDEv2(PreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "tildev2"

    def __init__(self, config: BertConfig, train_group_size=8):
        super().__init__(config)
        self.config = config
        self.bert = BertModel(config)
        self.tok_proj = torch.nn.Linear(config.hidden_size, 1)
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')
        self.train_group_size = train_group_size
        self.init_weights()

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def init_weights(self):
        self.bert.init_weights()
        self.tok_proj.apply(self._init_weights)

    def encode(self, **features):
        assert all([x in features for x in ['input_ids', 'attention_mask', 'token_type_ids']])
        model_out = self.bert(**features, return_dict=True)
        reps = self.tok_proj(model_out.last_hidden_state)
        tok_weights = torch.relu(reps)
        return tok_weights

    def forward(self, qry_in, doc_in):
        qry_input = qry_in
        doc_input = doc_in
        doc_out = self.bert(**doc_input, return_dict=True)
        doc_reps = self.tok_proj(doc_out.last_hidden_state)  # D * LD * d


        doc_reps = torch.relu(doc_reps) # relu to make sure no negative weights
        doc_input_ids = doc_input['input_ids']

        # mask ingredients
        qry_input_ids = qry_input['input_ids']
        qry_attention_mask = qry_input['attention_mask']
        self.mask_sep(qry_attention_mask)

        qry_reps = torch.ones_like(qry_input_ids, dtype=torch.float32, device=doc_reps.device).unsqueeze(2)
        tok_scores = self.compute_tok_score_cart(
            doc_reps, doc_input_ids,
            qry_reps, qry_input_ids, qry_attention_mask
        )  # Q * D

        scores = tok_scores

        labels = torch.arange(
            scores.size(0),
            device=doc_input['input_ids'].device,
            dtype=torch.long
        )

        # offset the labels
        labels = labels * self.train_group_size
        loss = self.cross_entropy(scores, labels)
        return loss, scores.view(-1)

    def mask_sep(self, qry_attention_mask):
        sep_pos = qry_attention_mask.sum(1).unsqueeze(1) - 1  # the sep token position
        _zeros = torch.zeros_like(sep_pos)
        qry_attention_mask.scatter_(1, sep_pos.long(), _zeros)
        return qry_attention_mask

    # This function credits to Luyu gao: https://github.com/luyug/COIL/blob/main/modeling.py
    def compute_tok_score_cart(self, doc_reps, doc_input_ids, qry_reps, qry_input_ids, qry_attention_mask):
        qry_input_ids = qry_input_ids.unsqueeze(2).unsqueeze(3)  # Q * LQ * 1 * 1
        doc_input_ids = doc_input_ids.unsqueeze(0).unsqueeze(1)  # 1 * 1 * D * LD
        exact_match = doc_input_ids == qry_input_ids  # Q * LQ * D * LD
        exact_match = exact_match.float()
        scores_no_masking = torch.matmul(
            qry_reps.view(-1, 1),  # (Q * LQ) * d
            doc_reps.view(-1, 1).transpose(0, 1)  # d * (D * LD)
        )
        scores_no_masking = scores_no_masking.view(
            *qry_reps.shape[:2], *doc_reps.shape[:2])  # Q * LQ * D * LD

        scores, _ = (scores_no_masking * exact_match).max(dim=3)  # Q * LQ * D, max pooling
        tok_scores = (scores * qry_attention_mask.unsqueeze(2))[:, 1:].sum(1)

        return tok_scores

class TILDEv2DocumentEncoder(DocumentEncoder):

    def __init__(self, model_name, device) -> None:
        self.model: TILDEv2 = TILDEv2.from_pretrained(model_name)
        self.device = device
        self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.reverse_voc = {v: k for k, v in self.tokenizer.vocab.items()}
    
    def encode(self, texts, **kwargs):
        p_max_len = kwargs.setdefault('p_max_len', 192)
        inputs = self.tokenizer(
            texts, 
            max_length=p_max_len, 
            padding='longest',
            truncation=True, 
            add_special_tokens=True,
            return_tensors='pt'
        ).to(self.device)
        term_weights = self.model.encode(**inputs).squeeze(-1).cpu().detach().numpy()
        return self._output_to_weight_dicts(inputs['input_ids'], term_weights)

    def _output_to_weight_dicts(self, batch_token_ids, batch_weights):
        to_return = []
        for i in range(len(batch_token_ids)):
            weights = batch_weights[i].flatten()
            tokens = self.tokenizer.convert_ids_to_tokens(batch_token_ids[i])
            tok_weights = {}
            for j in range(len(tokens)):
                tok = str(tokens[j])
                weight = float(weights[j])
                if tok == '[CLS]':
                    continue
                if tok == '[PAD]':
                    break
                if tok not in tok_weights:
                    tok_weights[tok] = weight
                elif weight > tok_weights[tok]:
                    tok_weights[tok] = weight
            to_return.append(tok_weights)
        return to_return


class TILDEv2QueryEncoder(QueryEncoder):

    def __init__(self, model_name_or_path, device='cpu'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        stop_ids = get_stop_ids(self.tokenizer)
        reverse_voc = {v: k for k, v in self.tokenizer.vocab.items()}
        self.stop_tokens = {reverse_voc[stop_id] for stop_id in stop_ids}

    def encode(self, text, **kwargs):
        tokens = self.tokenizer.tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_tokens]
        return dict(zip(tokens, [1] * len(tokens)))    
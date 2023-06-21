import torch
import torch.nn as nn
from itertools import accumulate
from transformers import AutoModel, AutoTokenizer
from .utils import clean_query, clean_passage

class DeepImpact(nn.Module):
    def __init__(self, model_name, device, max_seq_length=300):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.config = self.bert.config
        hidden_dropout_prob = self.config.hidden_dropout_prob if "hidden_dropout_prob" in dict(self.config) else \
            self.config.attention_dropout
        self.device = device
        self.max_seq_length = max_seq_length
        self.impact_score_encoder = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(768, 1),
            nn.ReLU()
        )
        self.init_weights()

    def convert_example(self, passage, max_seq_length):
        max_length = min(self.max_seq_length, max_seq_length)
        return self.tokenizer.encode_plus(passage, add_special_tokens=True, max_length=max_length, truncation=True, padding="max_length")

    def tokenize(self, query, passage):
        query_tokens = list(set(clean_query(query).strip().split()))
        content = clean_passage(passage).strip()
        doc_tokens = content.split()

        # NOTE: The following line accounts for CLS!
        tokenized = self.tokenizer.tokenize(content)
        word_indexes = list(accumulate([-1] + tokenized, lambda a, b: a + int(not b.startswith('##'))))
        match_indexes = list(set([doc_tokens.index(t) for t in query_tokens if t in doc_tokens]))
        term_indexes = [word_indexes.index(idx) for idx in match_indexes]

        a = [idx for i, idx in enumerate(match_indexes) if term_indexes[i] < self.max_seq_length]
        b = [idx for idx in term_indexes if idx < self.max_seq_length]

        return content, tokenized, a, b, len(word_indexes) + 2

    def forward(self, queries, passages):
        bsize = len(queries)
        pairs = []
        X, pfx_sum, pfx_sumX = [], [], []
        total_size, total_sizeX, max_seq_length = 0, 0, 0

        doc_partials = []
        pre_pairs = []

        for query, passage in zip(queries, passages):
            tokens, tokenized, term_idxs, token_idxs, seq_length = self.tokenize(query, passage)
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

    def encode(self, D, max_seq_length):
        if max_seq_length % 10 == 0:
            print("#>>>   max_seq_length = ", max_seq_length)

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
        term_scores = [[(term, y_score[pos]) for term, _, pos in terms] for terms in X]

        return term_scores
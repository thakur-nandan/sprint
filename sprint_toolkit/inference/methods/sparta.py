# Adapted from https://github.com/nreimers/beir-sparta/blob/main/SPARTA.py
# and https://github.com/nreimers/beir-sparta/blob/main/eval_msmarco.py

from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModel
from pyserini.encode import QueryEncoder, DocumentEncoder


class SPARTADocumentEncoder(torch.nn.Module, DocumentEncoder):
    def __init__(
        self, model_name, device
    ):  # SpanBERT/spanbert-base-cased'): #bert-base-uncased    #distilbert-base-uncased #distilroberta-base
        super().__init__()
        print("Model name:", model_name)
        self.bert_model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert_model.to(device)
        self.score_bias = torch.nn.Parameter(torch.tensor([1.0], device=device))
        self.device = device
        self.max_length = 300
        #####
        self.bert_input_emb = self.bert_model.embeddings.word_embeddings(
            torch.tensor(list(range(0, len(self.tokenizer))), device=device)
        )  # for building term weights
        self.reverse_voc = {v: k for k, v in self.tokenizer.vocab.items()}
        self.special_token_embedding_to_zero = False  # used during inference

    def bert_embeddings(self, input_ids):
        return self.bert_model.embeddings.word_embeddings(input_ids)

    def query_embeddings(self, query):
        queries_batch = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=self.max_length,
        ).to(self.device)
        queries_embeddings = self.bert_embeddings(queries_batch["input_ids"])
        return queries_embeddings

    def passage_embeddings(self, passages):
        passage_batch = self.tokenizer(
            passages,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        ).to(self.device)
        passage_embeddings = self.bert_model(**passage_batch).last_hidden_state
        return passage_embeddings

    def compute_scores(self, query_embeddings, passage_embeddings):
        ### Eq. 4 - Term matching
        scores = []
        for idx in range(
            len(query_embeddings)
        ):  # TODO: use correct pytorch function for this
            scores.append(
                torch.matmul(query_embeddings[idx], passage_embeddings.transpose(1, 2))
            )
        scores = torch.stack(scores)
        # print("Scores:", scores.shape)
        max_scores = torch.max(scores, dim=-1).values
        # print("Max-Scores:", max_scores.shape)

        ### Eq. 5 - ReLu
        relu_scores = torch.relu(
            max_scores
        )  # torch.relu(max_scores + self.score_bias)  #Bias score does not change that much?
        # print("ReLu-Scores:", relu_scores.shape)

        ### Eq. 6 - Final Score
        final_scores = torch.sum(
            torch.log(relu_scores + 1), dim=-1
        )  # .unsqueeze(dim=0)
        # print("Final scores:", final_scores.shape)
        return final_scores

    def forward(self, queries, passages):
        query_embeddings = self.query_embeddings(queries)
        passage_embeddings = self.passage_embeddings(passages)
        return self.compute_scores(query_embeddings, passage_embeddings)

    ###
    def _set_special_token_embedding_to_zero(self):
        if self.bert_model.training == True:
            return

        if self.special_token_embedding_to_zero:
            return

        for special_id in self.tokenizer.all_special_ids:
            self.bert_input_emb[special_id] = 0 * self.bert_input_emb[special_id]

        self.special_token_embedding_to_zero = True

    ###
    def encode(self, texts, **kwargs):
        self._set_special_token_embedding_to_zero()  # Important for full reproduction (although it seems to have little influence on the performance)

        term_weights_batch = []
        sparse_vec_size = kwargs.setdefault(
            "sparse_vec_size", 2000
        )  # TODO: Make this into the search.py cli arguments
        assert sparse_vec_size <= len(self.tokenizer)

        tokens = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt", max_length=500
        ).to(self.device)
        passage_embeddings = self.bert_model(**tokens).last_hidden_state
        for (
            passage_emb
        ) in passage_embeddings:  # TODO: Optimize this by batch operations
            scores = torch.matmul(self.bert_input_emb, passage_emb.transpose(0, 1))
            max_scores = torch.max(scores, dim=-1).values
            relu_scores = torch.relu(max_scores)  # Eq. 5
            final_scores = torch.log(relu_scores + 1)  # Eq. 6, final score

            top_results = torch.topk(final_scores, k=sparse_vec_size)
            tids = top_results[1].cpu().detach().tolist()
            scores = top_results[0].cpu().detach().tolist()

            term_weights = {}
            for tid, score in zip(tids, scores):
                if score > 0:
                    term_weights[self.reverse_voc[tid]] = score
                else:
                    break

            term_weights_batch.append(term_weights)

        return term_weights_batch


class SPARTAQueryEncoder(QueryEncoder):
    def __init__(self, model_name_or_path, device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.reverse_voc = {v: k for k, v in self.tokenizer.vocab.items()}

    def encode(self, text, **kwargs):
        token_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        tokens = [self.reverse_voc[token_id] for token_id in token_ids]
        term_weights = defaultdict(int)

        # Important for reproducing the results:
        # Note that in Pyserini/Anserini, the query term weights are maintained by JHashMap,
        # which will keep only one term weight for identical terms
        for token in tokens:
            term_weights[token] += 1
        return term_weights

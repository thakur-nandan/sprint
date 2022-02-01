import torch
from transformers import AutoTokenizer, AutoModel

class SPARTA(torch.nn.Module):
    def __init__(self, model_name, device): #SpanBERT/spanbert-base-cased'): #bert-base-uncased    #distilbert-base-uncased #distilroberta-base
        super().__init__()
        print("Model name:", model_name)
        self.bert_model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert_model.to(device)
        self.score_bias = torch.nn.Parameter(torch.tensor([1.0], device=device))
        self.device = device
        self.max_length = 300

    def bert_embeddings(self, input_ids):
        return self.bert_model.embeddings.word_embeddings(input_ids)

    def query_embeddings(self, query):
        queries_batch = self.tokenizer(query, padding=True, truncation=True, return_tensors='pt', add_special_tokens=False, max_length=self.max_length).to(self.device)
        queries_embeddings = self.bert_embeddings(queries_batch['input_ids'])
        return queries_embeddings

    def passage_embeddings(self, passages):
        passage_batch = self.tokenizer(passages, padding=True, truncation=True, return_tensors='pt', max_length=self.max_length).to(self.device)
        passage_embeddings = self.bert_model(**passage_batch).last_hidden_state
        return passage_embeddings

    def compute_scores(self, query_embeddings, passage_embeddings):
        ### Eq. 4 - Term matching
        scores = []
        for idx in range(len(query_embeddings)):        #TODO: use correct pytorch function for this
            scores.append(torch.matmul(query_embeddings[idx], passage_embeddings.transpose(1, 2)))
        scores = torch.stack(scores)
        #print("Scores:", scores.shape)
        max_scores = torch.max(scores, dim=-1).values
        #print("Max-Scores:", max_scores.shape)

        ### Eq. 5 - ReLu
        relu_scores = torch.relu(max_scores)    #torch.relu(max_scores + self.score_bias)  #Bias score does not change that much?
        #print("ReLu-Scores:", relu_scores.shape)

        ### Eq. 6 - Final Score
        final_scores = torch.sum(torch.log(relu_scores + 1), dim=-1) #.unsqueeze(dim=0)
        #print("Final scores:", final_scores.shape)
        return final_scores

    def forward(self, queries, passages):
        query_embeddings = self.query_embeddings(queries)
        passage_embeddings = self.passage_embeddings(passages)
        return self.compute_scores(query_embeddings, passage_embeddings)
import torch
from torch import nn, Tensor
from typing import Iterable, Dict
from .utils import pairwise_dot_score

class FLOPS:
    """constraint from Minimizing FLOPs to Learn Efficient Sparse Representations
    https://arxiv.org/abs/2004.05665
    """
    def __call__(self, batch_rep):
        return torch.sum(torch.mean(torch.abs(batch_rep), dim=0) ** 2)

class MarginMSELossSplade(nn.Module):
    """
    Compute the MSE loss between the |sim(Query, Pos) - sim(Query, Neg)| and |gold_sim(Q, Pos) - gold_sim(Query, Neg)|
    By default, sim() is the dot-product
    For more details, please refer to https://arxiv.org/abs/2010.02666
    """
    def __init__(self, model, similarity_fct = pairwise_dot_score, lambda_d=8e-2, lambda_q=1e-1):
        """
        :param model: SentenceTransformerModel
        :param similarity_fct:  Which similarity function to use
        """
        super(MarginMSELossSplade, self).__init__()
        self.model = model
        self.similarity_fct = similarity_fct
        self.loss_fct = nn.MSELoss()
        self.lambda_d = lambda_d
        self.lambda_q = lambda_q
        self.FLOPS = FLOPS()

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        # sentence_features: query, positive passage, negative passage
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        embeddings_query = reps[0]
        embeddings_pos = reps[1]
        embeddings_neg = reps[2]

        scores_pos = self.similarity_fct(embeddings_query, embeddings_pos)
        scores_neg = self.similarity_fct(embeddings_query, embeddings_neg)
        margin_pred = scores_pos - scores_neg

        flops_doc = self.lambda_d*(self.FLOPS(embeddings_pos) + self.FLOPS(embeddings_neg))
        flops_query = self.lambda_q*(self.FLOPS(embeddings_query))

        return self.loss_fct(margin_pred, labels) + flops_doc + flops_query
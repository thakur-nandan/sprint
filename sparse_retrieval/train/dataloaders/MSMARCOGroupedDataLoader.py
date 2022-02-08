import random
import re
import datasets
from typing import Union, List

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from nltk.corpus import stopwords

class GroupedMarcoTrainDataset(Dataset):
    query_columns = ['qid', 'query']
    document_columns = ['pid', 'passage']

    def __init__(
            self,
            path_to_tsv: Union[List[str], str],
            tokenizer: PreTrainedTokenizer,
            q_max_len: int,
            p_max_len: int,
            train_group_size: int,
            cache_dir: str,
            stopwords: bool = False,
    ):
        self.tok = tokenizer
        self.SEP = [self.tok.sep_token_id]
        self.stop_ids = self.get_stop_ids() if stopwords else set()
        self.q_max_len = q_max_len
        self.p_max_len = p_max_len
        self.train_group_size = train_group_size

        self.nlp_dataset = datasets.load_dataset(
            'json',
            cache_dir=cache_dir,
            data_files=path_to_tsv,
            ignore_verifications=False,
            features=datasets.Features({
                'qry': {
                    'qid': datasets.Value('string'),
                    'query': [datasets.Value('int32')],
                },
                'pos': [{
                    'pid': datasets.Value('string'),
                    'passage': [datasets.Value('int32')],
                }],
                'neg': [{
                    'pid': datasets.Value('string'),
                    'passage': [datasets.Value('int32')],
                }]}
            )
        )['train']
        self.total_len = len(self.nlp_dataset)

    def create_one_example(self, text_encoding, is_query=False):
        item = self.tok.encode_plus(
            text_encoding,
            truncation='only_first',
            return_attention_mask=False,
            max_length=self.q_max_len if is_query else self.p_max_len,
        )
        return item

    def __len__(self):
        return self.total_len

    def get_stop_ids(self):
        stop_words = set(stopwords.words('english'))
        # keep some common words in ms marco questions
        stop_words.difference_update(["where", "how", "what", "when", "which", "why", "who"])

        vocab = self.tok.get_vocab()
        tokens = vocab.keys()

        stop_ids = []

        for stop_word in stop_words:
            ids = self.tok(stop_word, add_special_tokens=False)["input_ids"]
            if len(ids) == 1:
                # bad_token.append(stop_word)
                stop_ids.append(ids[0])

        for token in tokens:
            token_id = vocab[token]
            if token_id in stop_ids:
                continue
            if token == '##s':
                stop_ids.append(token_id)
            if token[0] == '#' and len(token) > 1:
                continue
            if not re.match("^[A-Za-z0-9_-]*$", token):
                # bad_token.append(token)
                stop_ids.append(token_id)

        return set(stop_ids)

    def __getitem__(self, item):
        group = self.nlp_dataset[item]
        group_batch = []
        qid, qry = (group['qry'][k] for k in self.query_columns)
        qry = [id for id in qry if id not in self.stop_ids]

        encoded_query = self.create_one_example(qry, is_query=True)
        _, pos_psg = [
            random.choice(group['pos'])[k] for k in self.document_columns]
        group_batch.append(self.create_one_example(pos_psg))
        if len(group['neg']) < self.train_group_size - 1:
            negs = random.choices(group['neg'], k=self.train_group_size - 1)
        else:
            negs = random.sample(group['neg'], k=self.train_group_size - 1)
        for neg_entry in negs:
            _, neg_psg = [neg_entry[k] for k in self.document_columns]
            group_batch.append(self.create_one_example(neg_psg))

        return encoded_query, group_batch

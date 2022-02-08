import os
from dataclasses import dataclass, field
from typing import Union, List

@dataclass
class DataArguments:
    train_dir: str = field(
        default=None, metadata={"help": "Path to train directory"}
    )
    train_path: Union[str] = field(
        default=None, metadata={"help": "Path to train data"}
    )
    train_group_size: int = field(default=8)

    pred_path: List[str] = field(default=None, metadata={"help": "Path to prediction data"})
    pred_dir: str = field(
        default=None, metadata={"help": "Path to prediction directory"}
    )
    pred_id_file: str = field(default=None)
    rank_score_path: str = field(default=None, metadata={"help": "where to save the match score"})

    encode_in_path: List[str] = field(default=None, metadata={"help": "Path to data to encode"})
    encoded_save_path: str = field(default=None, metadata={"help": "where to save the encode"})

    q_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for query. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    p_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    document: bool = field(default=False)

    def __post_init__(self):
        if self.train_dir is not None:
            files = os.listdir(self.train_dir)
            self.train_path = [
                os.path.join(self.train_dir, f)
                for f in files
                if f.endswith('tsv') or f.endswith('json')
            ]
        if self.pred_dir is not None:
            files = os.listdir(self.pred_dir)
            self.pred_path = [
                os.path.join(self.pred_dir, f)
                for f in files
            ]
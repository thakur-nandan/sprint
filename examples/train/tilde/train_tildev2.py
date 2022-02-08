from transformers import AutoConfig, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, HfArgumentParser
from sparse_retrieval.train import models, SPARSETrainer
from sparse_retrieval.train.dataloaders import GroupedMarcoTrainDatasetTILDE
from torch import Tensor
from dataclasses import dataclass, field
from typing import Dict
import os
from typing import Dict

@dataclass
class TILDEv2TrainingArguments(TrainingArguments):
    warmup_ratio: float = field(default=0)
    model_name: str = field(default='bert-base-uncased')
    q_max_len: int = field(default=16)
    p_max_len: int = field(default=192)
    train_group_size: int = field(default=8)
    train_dir: str = field(default=None)
    cache_dir: str = field(default='./cache')
    report_to = []

    def __post_init__(self):
        files = os.listdir(self.train_dir)
        self.train_path = [
            os.path.join(self.train_dir, f)
            for f in files
            if f.endswith('tsv') or f.endswith('json')
        ]

@dataclass
class QryDocCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_q_len: int = 16
    max_d_len: int = 192

    def __call__(self, features) -> Dict[str, Dict[str, Tensor]]:
        qq = [f[0] for f in features]
        dd = [f[1] for f in features]
        if isinstance(dd[0], list):
            dd = sum(dd, [])
        q_collated = self.tokenizer.pad(
            qq,
            padding='max_length',
            max_length=self.max_q_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer.pad(
            dd,
            padding='max_length',
            max_length=self.max_d_len,
            return_tensors="pt",
        )

        return {'qry_in': q_collated, 'doc_in': d_collated}


def main():
    parser = HfArgumentParser(TILDEv2TrainingArguments)
    args: TILDEv2TrainingArguments = parser.parse_args_into_dataclasses()[0]
    config = AutoConfig.from_pretrained(args.model_name,
                                        num_labels=1,
                                        cache_dir=args.cache_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name,
                                              cache_dir=args.cache_dir,
                                              use_fast=False)

    model = models.TILDEv2.from_pretrained(args.model_name, config=config,
                                    train_group_size=args.train_group_size,
                                    cache_dir=args.cache_dir)

    train_dataset = GroupedMarcoTrainDatasetTILDE(path_to_tsv=args.train_path,
                                             p_max_len=args.p_max_len,
                                             q_max_len=args.q_max_len,
                                             tokenizer=tokenizer,
                                             train_group_size=args.train_group_size,
                                             cache_dir=args.cache_dir)

    trainer = SPARSETrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=QryDocCollator(tokenizer, max_q_len=args.q_max_len, max_d_len=args.p_max_len),
    )

    # Training
    trainer.train()
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
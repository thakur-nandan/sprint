import shutil
import pytest
import tempfile
import os


@pytest.fixture(name="bert_path", scope="session")
def bert_path_fixture() -> str:
    try:
        local_dir = tempfile.mkdtemp()

        import torch

        torch.manual_seed(42)
        vocab = [
            "[PAD]",
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[MASK]",
            "the",
            "of",
            "and",
            "in",
            "to",
            "was",
            "he",
        ]
        vocab_file = os.path.join(local_dir, "vocab.txt")
        with open(vocab_file, "w") as f:
            f.write("\n".join(vocab))

        from transformers import BertConfig, BertModel, BertTokenizer

        config = BertConfig(
            vocab_size=len(vocab),
            hidden_size=2,
            num_attention_heads=1,
            num_hidden_layers=2,
            intermediate_size=2,
            max_position_embeddings=512,
        )

        bert = BertModel(config)
        tokenizer = BertTokenizer(vocab_file)

        bert.save_pretrained(local_dir)
        tokenizer.save_pretrained(local_dir)

        yield local_dir
    finally:
        shutil.rmtree(local_dir)
        print("Cleared temporary DistilBERT model")


@pytest.fixture(name="distilbert_path", scope="session")
def distilbert_path_fixture() -> str:
    try:
        local_dir = tempfile.mkdtemp()

        import torch

        torch.manual_seed(42)
        vocab = [
            "[PAD]",
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[MASK]",
            "the",
            "of",
            "and",
            "in",
            "to",
            "was",
            "he",
        ]
        vocab_file = os.path.join(local_dir, "vocab.txt")
        with open(vocab_file, "w") as f:
            f.write("\n".join(vocab))

        from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizer

        config = DistilBertConfig(
            vocab_size=len(vocab),
            hidden_size=2,
            num_attention_heads=1,
            num_hidden_layers=2,
            intermediate_size=2,
            max_position_embeddings=512,
        )

        bert = DistilBertModel(config)
        tokenizer = DistilBertTokenizer(vocab_file)

        bert.save_pretrained(local_dir)
        tokenizer.save_pretrained(local_dir)

        yield local_dir
    finally:
        shutil.rmtree(local_dir)
        print("Cleared temporary DistilBERT model")


@pytest.fixture(name="scifact_path", scope="session")
def scifact_path_fixture() -> str:
    return "sample-data/scifact"

import os
import shutil
import pytest
from sprint.inference import aio


@pytest.mark.parametrize(
    "ckpt_name, encoder_name",
    [
        ("bert_path", "unicoil"),
        ("distilbert_path", "splade"),
        ("distilbert_path", "sparta"),
        ("bert_path", "deepimpact"),
    ],
)
def test_aio(
    ckpt_name: str, encoder_name: str, scifact_path: str, request: pytest.FixtureRequest
) -> None:
    ckpt_name = request.getfixturevalue(ckpt_name)
    output_dir = "pytest-output"
    output_quantized_dir = "pytest-output-quantized"
    try:
        aio.run(
            encoder_name=encoder_name,
            ckpt_name=ckpt_name,
            data_name="beir/scifact",
            train_data_dir=scifact_path,
            eval_data_dir=scifact_path,
            gpus=["cpu"],
            output_dir=output_dir,
            do_quantization=True,
            quantization_method="range-nbits",  # So the doc term weights will be quantized by `(term_weights / 5) * (2 ** 8)`
            original_score_range=5,
            quantization_nbits=8,
            original_query_format="beir",
            topic_split="test",
        )
        # You would get "NDCG@10": 0.68563
    finally:
        for dir in [output_dir, output_quantized_dir]:
            if os.path.exists(dir):
                shutil.rmtree(dir)


# TODO: Add reranking tests

from builtins import help
from typing import List

from . import encode
from . import quantize
from . import index
from . import reformat_query
from . import search
from . import evaluate
import argparse
import logging
import os


logger = logging.getLogger(__name__)


def run(
    # encode
    encoder_name: str,
    ckpt_name: List[str],
    data_name: str,
    gpus: List[int],
    output_dir: str,
    # reformat_query
    original_query_format: str,
    # quantize
    do_quantization: bool,
    quantization_method: str = None,  # TODO: Merge `quantization_method`` and `do_quantization``
    original_score_range: float = None,
    quantization_nbits: int = None,
    ndigits: int = None,
    min_idf: int = -1,
    # search
    topic_split: str = "test",
    hits: int = 1000,
    output_format_search: str = "trec",
    # evaluate
    k_values: List[int] = [1, 2, 3, 5, 10, 20, 100, 1000],
    # default setting
    batch_size: int = 64,
    chunk_size: int = 100000,
    nprocs: int = 12,
    data_dir: str = None,
    train_data_dir: str = None,
    eval_data_dir: str = None,
):
    # 0. Check if the data for training is in the same directory as eval
    # i.e., if you provide data_dir, it means both training and evaluation corpus is present together.
    if data_dir:
        train_data_dir, eval_data_dir = data_dir, data_dir

    # Check if the same checkpoint is used for both doc and query
    if type(ckpt_name) == str:
        query_ckpt, doc_ckpt = ckpt_name, ckpt_name
    elif type(ckpt_name) == list:
        if len(ckpt_name) == 1:
            query_ckpt, doc_ckpt = ckpt_name[0], ckpt_name[0]
        else:
            query_ckpt, doc_ckpt = ckpt_name[0], ckpt_name[1]

    # 1. Encode the documents into term weights
    # The output will be ${output_dir}/collection
    output_dir_encode = os.path.join(output_dir, "collection")
    if not os.path.exists(output_dir_encode):
        encode.run(
            encoder_name,
            doc_ckpt,
            data_name,
            train_data_dir,
            gpus,
            output_dir_encode,
            batch_size,
            chunk_size,
        )
    else:
        logger.info("Escaped encoding due to the existing output file(s)")

    # 2. Quantize the term weights from floats into integers
    # The output will be ${output_dir}-quantized/collection, if `do_quantization == True`
    collection_dir = os.path.join(output_dir, "collection")
    output_dir_quantize = None
    if do_quantization:
        output_dir += "-quantized"  # TODO: Change this into a specific name
        output_dir_quantize = os.path.join(output_dir, "collection")
        if not os.path.exists(output_dir_quantize):
            quantize.run(
                collection_dir,
                output_dir_quantize,
                quantization_method,
                original_score_range,
                quantization_nbits,
                ndigits,
                nprocs,
            )
        else:
            logger.info("Escaped quantization due to the existing output file(s)")
    else:
        logger.info("Escaped quantization due to `do_quantization == False`")

    if do_quantization:
        collection_dir = output_dir_quantize

    # 3. Index the term weights into a Lucene-format index file
    # The output will be ${output_dir}-quantized/index
    output_dir_index = os.path.join(output_dir, "index")
    if not os.path.exists(output_dir_index):
        index.run(
            "JsonVectorCollection",
            collection_dir,
            output_dir_index,
            "DefaultLuceneDocumentGenerator",
            True,
            True,
            nprocs,
        )
    else:
        logger.info("Escaped indexing due to the existing output file(s)")

    # 4. Reformat the queries into Pyserini-compatible
    # The output will be under the data directory by default
    if (
        "beir/" in data_name or "beir_" in data_name
    ) and eval_data_dir is None:  # TODO: Unify this along with the same snippet within data_iters.py
        eval_data_dir = os.path.join("datasets", data_name.replace("beir_", "beir/"))
    tsv_queries_path = os.path.join(eval_data_dir, f"queries-{topic_split}.tsv")
    if not os.path.exists(tsv_queries_path):
        reformat_query.run(original_query_format, eval_data_dir, topic_split, None)
    else:
        logger.info("Escaped reformatting quries due to the existing output file(s)")

    # 5. Search the queries over the index
    # The output will be ${output_dir}-quantized/${output_format_search}-format/run.tsv
    output_path_search = os.path.join(
        output_dir, f"{output_format_search}-format/run.tsv"
    )
    if not os.path.exists(output_path_search):
        search.run(
            topics=tsv_queries_path,
            encoder_name=encoder_name,
            ckpt_name=query_ckpt,
            index=output_dir_index,
            output=output_path_search,
            impact=True,
            hits=hits + 1,
            batch_size=batch_size,
            threads=nprocs,
            output_format=output_format_search,
            min_idf=min_idf,
        )
    else:
        logger.info("Escaped search due to the existing output file(s)")

    # 6. Do evaluation over the results generated by the last step
    # The output will be under ${output_dir}-quantized/evaluation
    qrels_path = os.path.join(eval_data_dir, "qrels", f"{topic_split}.tsv")
    output_dir_evaluate = os.path.join(output_dir, "evaluation")
    if not os.path.exists(output_dir_evaluate):
        evaluate.run(
            output_path_search,
            output_format_search,
            qrels_path,
            output_dir_evaluate,
            k_values,
        )
    else:
        logger.info("Escaped evaluation due to the existing output file(s)")

    logger.info(f"{__name__}: Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_name")
    parser.add_argument(
        "--ckpt_name",
        nargs="+",
        type=str,
        help="Checkpoint name, can be only string or pass query and document checkpoints respectively",
    )
    parser.add_argument("--data_name")
    parser.add_argument("--data_dir", required=False)
    parser.add_argument("--eval_data_dir", required=False)
    parser.add_argument("--train_data_dir", required=False)
    parser.add_argument("--gpus", nargs="+", type=int)
    parser.add_argument("--output_dir")

    parser.add_argument(
        "--min_idf",
        type=int,
        default=-1,
        help="Query tokens with IDF <= this value will be ignored. The default value of -1 means it considers all the tokens",
    )
    parser.add_argument("--do_quantization", action="store_true")
    parser.add_argument(
        "--quantization_method",
        required=False,
        choices=["range-nbits", "ndigits-round"],
    )
    # params for 'range-nbits' quantization technique:
    parser.add_argument("--original_score_range", type=float, default=5)
    parser.add_argument("--quantization_nbits", type=int, default=8)
    # params for 'ndigits-round' quantization technique:
    parser.add_argument("--ndigits", type=int, default=2, help="2 means *100")
    # param: query format whether it is in msmarco or beir format?
    parser.add_argument("--original_query_format", help="e.g. beir")
    # params for evaluation
    parser.add_argument(
        "--topic_split",
        choices=["train", "test", "dev"],
        help="The queries split for search and evaluation",
    )
    parser.add_argument("--hits", type=int, default=1000)
    parser.add_argument(
        "--output_format_search", type=str, default="trec", choices=["msmarco", "trec"]
    )

    parser.add_argument(
        "--k_values", nargs="+", type=int, default=[1, 2, 3, 5, 10, 20, 100, 1000]
    )

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--chunk_size", type=int, default=100000)
    parser.add_argument("--nprocs", type=int, default=12)

    args = parser.parse_args()
    run(**vars(args))

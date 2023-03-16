from sprint.inference import aio
import argparse

if __name__ == '__main__':  # aio.run can only be called within __main__

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, default="castorini/unicoil-msmarco-passage")
    parser.add_argument("--output_results_path", type=str, default=None)
    parser.add_argument("--train_data_dir", type=str, default=None)
    parser.add_argument("--eval_data_dir", type=str, default=None)
    parser.add_argument('--gpus', nargs='+', type=int)
    args = parser.parse_args()

    aio.run(
        encoder_name='unicoil',
        ckpt_name=args.model_name_or_path,
        data_name='beir/{}'.format(args.dataset),
        train_data_dir=args.train_data_dir,
        eval_data_dir=args.eval_data_dir,
        gpus=args.gpus,
        output_dir=args.output_results_path,
        do_quantization=True,
        quantization_method='range-nbits',  # So the doc term weights will be quantized by `(term_weights / 5) * (2 ** 8)`
        original_score_range=5,
        quantization_nbits=8,
        original_query_format='beir',
        topic_split=args.split
    )
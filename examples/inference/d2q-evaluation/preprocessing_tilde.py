from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
import logging, sys
import os, json
import tqdm
import argparse
import pathlib

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

def load_d2q_expansion(filepath):
    logging.info("Loading doc2query expansions for uniCOIL...")
    expansions = {}
    num_lines = sum(1 for i in open(filepath, 'rb'))
    with open(filepath, encoding='utf8') as fIn:
        for line in tqdm.tqdm(fIn, total=num_lines):
            line = json.loads(line)
            expansions[line.get("pid")] = {
                "expansion": line.get("psg", ""),
            } 
    return expansions

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--d2q_filepath", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=False)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--dataset_dir", type=str, default=None)
    args = parser.parse_args()

    dataset = args.dataset
    data_path = args.dataset_dir
    out_path = args.output_path
    d2q_filepath = args.d2q_filepath

    # Making the output directory for storing corpus.jsonl file for uniCOIL
    if out_path:
        os.makedirs(out_path, exist_ok=True)

    # Loading original BEIR corpora and splits
    if not data_path:
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
        out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
        data_path = util.download_and_unzip(url, out_dir)
        corpus = GenericDataLoader(data_folder=data_path).load_corpus()
    
    elif data_path:
        corpus = GenericDataLoader(os.path.join(data_path)).load_corpus()
    
    # Loading Expansions file
    expansions = load_d2q_expansion(d2q_filepath)

    logging.info("Preprocessing Corpus for TILDE docT5query...")
    output_corpus = os.path.join(out_path, "corpus.jsonl")

    with open(output_corpus, 'w') as fOut:
        for doc_id in tqdm.tqdm(corpus, total=len(corpus)):
            title = corpus[doc_id].get("title", "")
            document = corpus[doc_id].get("text", "")
            expansion = expansions[doc_id].get("expansion", "")
            json.dump({
                "id": doc_id, 
                "contents": document + "\n" + expansion,
                "title": title
            }, fOut)
            fOut.write('\n')

if __name__ == "__main__":
    main()

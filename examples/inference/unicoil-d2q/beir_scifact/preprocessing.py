from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from transformers import AutoTokenizer
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
            expansions[line.get("id")] = {
                "queries": " ".join(line.get("queries", [])),
            } 
    return expansions

def preprocess_expansion(document, expansion, tokenizer, 
                         doc_max_length=400, expansion_max_length=100):

        try:
            doc_tids = tokenizer.encode(document, add_special_tokens=False, max_length=doc_max_length, truncation=True)
            expansion_tids = tokenizer.encode(expansion, add_special_tokens=False, max_length=expansion_max_length, truncation=True)
            injects = set()
            
            for exp_tid in expansion_tids:
                if exp_tid not in doc_tids:
                    injects.add(exp_tid)
            
            all_tids = doc_tids + [tokenizer.sep_token_id] + list(injects)  # important here: do not add special tokens at the two ends!!!
            doc_and_expansion = tokenizer.decode(all_tids)  # important here: use decode but convert_ids_to_tokens!!!
            return doc_and_expansion
        
        except:
            if len(document) <= 1: #edge case: where document is empty (random queries are generated!)
                return ""

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--d2q_filepath", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=False)
    parser.add_argument("--model_name_or_path", type=str, default="castorini/unicoil-msmarco-passage")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--dataset_dir", type=str, default=None)
    args = parser.parse_args()

    dataset = args.dataset
    data_path = args.dataset_dir
    model_name_or_path = args.model_name_or_path
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

    # Provide the uniCOIL Tokenizer: (castorini/unicoil-msmarco-passage)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    
    # Loading Expansions file
    expansions = load_d2q_expansion(d2q_filepath)

    logging.info("Preprocessing Corpus for uniCOIL...")
    output_corpus = os.path.join(out_path, "corpus.jsonl")

    with open(output_corpus, 'w') as fOut:
        for doc_id in tqdm.tqdm(corpus, total=len(corpus)):
            document = corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "")
            expansion = expansions[doc_id].get("queries", "")
            output_text = preprocess_expansion(document, expansion, tokenizer)
            json.dump({
                "_id": doc_id, 
                "text": output_text,
                "title": ""
            }, fOut)
            fOut.write('\n')

if __name__ == "__main__":
    main()

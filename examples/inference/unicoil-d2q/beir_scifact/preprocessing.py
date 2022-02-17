from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from transformers import AutoTokenizer
from distutils.dir_util import copy_tree
import logging, sys
import os, json
import shutil
import tqdm


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


dataset = sys.argv[1]
data_path = sys.argv[2]
split = sys.argv[3]
model_name_or_path = sys.argv[4]
out_path = sys.argv[5]

if not os.path.isdir(out_path):
    logging.info("Found path empty. Making new directory: {}".format(out_path))
    os.makedirs(out_path)
    logging.info("Copying: {} to output path: {}".format(data_path, out_path))
    copy_tree(data_path, out_path)

else:
    logging.info("Deleting existing path: {}".format(out_path))
    shutil.rmtree(out_path)
    logging.info("Copying: {} to output path: {}".format(data_path, out_path))
    copy_tree(data_path, out_path)

corpus, queries, qrels = GenericDataLoader(data_path).load(split=split)
corpus_orig = GenericDataLoader("/home/ukp/thakur/projects/sbert_retriever/datasets-new/{}".format(dataset)).load_corpus()

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

def preprocess_expansion(doc_id, doc_and_expansion, tokenizer, sep='\n', doc_max_length=400, expansion_max_length=100):
        try:
            doc, expansion = doc_and_expansion.strip().split(sep)
            doc_tids = tokenizer.encode(doc, add_special_tokens=False, max_length=doc_max_length, truncation=True)
            expansion_tids = tokenizer.encode(expansion, add_special_tokens=False, max_length=expansion_max_length, truncation=True)
            injects = set()
            for exp_tid in expansion_tids:
                if exp_tid not in doc_tids:
                    injects.add(exp_tid)
            all_tids = doc_tids + [tokenizer.sep_token_id] + list(injects)  # important here: do not add special tokens at the two ends!!!
            doc_and_expansion = tokenizer.decode(all_tids)  # important here: use decode but convert_ids_to_tokens!!!
            return doc_and_expansion
        
        except:
            if len(doc_and_expansion.strip().split(sep)) == 1: #edge case: where document is empty (random queries are generated!)
                return ""
        
output_file = os.path.join(out_path, "corpus.jsonl")

logging.info("Preprocessing Corpus for uniCOIL...")

with open(output_file, 'w') as fOut:
    for doc_id in tqdm.tqdm(corpus, total=len(corpus)):
        input_text = corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "")
        output_text = preprocess_expansion(doc_id, input_text, tokenizer)
        json.dump({
            "_id": doc_id, 
            "text": output_text,
            "title": ""
        }, fOut)
        fOut.write('\n')
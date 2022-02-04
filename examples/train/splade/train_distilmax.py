from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from sparse_retrieval.train.dataloaders import MSMARCODataset 
from sparse_retrieval.train import models, losses

from datetime import datetime
import logging
import sys, os
import json
import gzip
import tqdm
import argparse
import pathlib

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

parser = argparse.ArgumentParser()
parser.add_argument("--train_batch_size", default=64, type=int)
parser.add_argument("--max_seq_length", default=256, type=int)
parser.add_argument("--model_name", default="distilbert-base-uncased", type=str)
parser.add_argument("--max_passages", default=0, type=int)
parser.add_argument("--epochs", default=30, type=int)
parser.add_argument("--negs_to_use", default=None, help="From which systems should negatives be used ? Multiple systems seperated by comma. None = all")
parser.add_argument("--warmup_steps", default=1000, type=int)
parser.add_argument("--lambda_d", default=0.08, type=float)
parser.add_argument("--lambda_q", default=0.1, type=float)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--num_negs_per_system", default=5, type=int)
parser.add_argument("--use_all_queries", default=False, action="store_true")
args = parser.parse_args()

logging.info(str(args))

#################################
#### Parameters for Training ####
#################################
train_batch_size = args.train_batch_size  # Increasing the train batch size generally improves the model performance, but requires more GPU memory
model_name = args.model_name
max_passages = args.max_passages
max_seq_length = args.max_seq_length  # Max length for passages. Increasing it implies more GPU memory needed
num_negs_per_system = args.num_negs_per_system  # We used different systems to mine hard negatives. Number of hard negatives to add from each system
num_epochs = args.epochs  # Number of epochs we want to train

logging.info("Create new SPLADE (Distilmax) model")
word_embedding_model = models.MLMTransformer(model_name, max_seq_length=max_seq_length)
model = SentenceTransformer(modules=[word_embedding_model])

#### Provide model save path
model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "output", "distilSplade-q-{}-d-{}-{}".format(args.lambda_q, args.lambda_d, model_name.replace("/", "-")))
os.makedirs(model_save_path, exist_ok=True)

#############################
#### Load MSMARCO (BEIR) ####
#############################

#### Download msmarco.zip dataset and unzip the dataset
dataset = "msmarco"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

### Load BEIR MSMARCO training dataset, this will be used for query and corpus for reference.
corpus, queries, _ = GenericDataLoader(data_path).load(split="train")

##################################################
#### Download MSMARCO Hard Negs Triplets File ####
##################################################

triplets_url = "https://sbert.net/datasets/msmarco-hard-negatives.jsonl.gz"
msmarco_triplets_filepath = os.path.join(data_path, "msmarco-hard-negatives.jsonl.gz")
if not os.path.isfile(msmarco_triplets_filepath):
    util.download_url(triplets_url, msmarco_triplets_filepath)

#### Load the hard negative MSMARCO jsonl triplets from SBERT 
#### These contain a ce-score which denotes the cross-encoder score for the query and passage.
logging.info("Loading MSMARCO hard-negatives...")

train_queries = {}
with gzip.open(msmarco_triplets_filepath, 'rt', encoding='utf8') as fIn:
    for line in tqdm.tqdm(fIn, total=502939):
        data = json.loads(line)
        
        #Get the positive passage ids
        pos_pids = [item['pid'] for item in data['pos']]
        pos_scores = dict(zip(pos_pids, [item['ce-score'] for item in data['pos']]))
        
        #Get all the hard negatives
        neg_pids = set()
        neg_scores = {}
        for system_negs in data['neg'].values():
            for item in system_negs:
                pid = item['pid']
                score = item['ce-score']
                if pid not in neg_pids:
                    neg_pids.add(pid)
                    neg_scores[pid] = score
        
        if len(pos_pids) > 0 and len(neg_pids) > 0:
            train_queries[data['qid']] = {'query': queries[data['qid']], 'pos': pos_pids, 'pos_scores': pos_scores, 
                                          'hard_neg': neg_pids, 'hard_neg_scores': neg_scores}
        
logging.info("Train queries: {}".format(len(train_queries)))

# For training the SentenceTransformer model, we need a dataset, a dataloader, and a loss used for training.
train_dataset = MSMARCODataset(queries=train_queries, corpus=corpus)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size, drop_last=True)
train_loss = losses.MarginMSELossSplade(model=model, lambda_d=args.lambda_d, lambda_q=args.lambda_q)

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          warmup_steps=args.warmup_steps,
          use_amp=True,
          checkpoint_path=model_save_path,
          checkpoint_save_steps=10000,
          optimizer_params = {'lr': args.lr})

# Save model
model.save(model_save_path)
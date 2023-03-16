from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from sprint.train import models

from torch.cuda.amp import autocast
import transformers
import torch
import logging
import pathlib
import os
import argparse
import math
import gzip
import random
import tqdm
import json

random.seed(42)

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

parser = argparse.ArgumentParser()
parser.add_argument("--train_batch_size", default=4, type=int)
parser.add_argument("--max_seq_length", default=256, type=int)
parser.add_argument("--model_name", default="distilbert-base-uncased", type=str)
parser.add_argument("--max_passages", default=0, type=int)
parser.add_argument("--epochs", default=1, type=int)
parser.add_argument("--warmup_steps", default=1000, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--num_negatives", default=35, type=int)
parser.add_argument("--grad_acc_steps", default=1, type=int)
parser.add_argument("--max_grad_norm", default=1, type=float)
args = parser.parse_args()

logging.info(str(args))

#################################
#### Parameters for Training ####
#################################
model_name = args.model_name
num_epochs = args.epochs
lr = args.lr
grad_acc_steps = args.grad_acc_steps
batch_size = args.train_batch_size
num_negatives = args.num_negatives
max_grad_norm = args.max_grad_norm

logging.info(f"model_name: {model_name}")
logging.info(f"batch_size: {batch_size}")
logging.info(f"num_neg: {num_negatives}")

# Loading SPARTA Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = models.DeepImpact(model_name, device)
model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "output", "DeepImpact-{}".format(model_name.replace("/", "-")))
model.tokenizer.save_pretrained(model_save_path)

#############################
#### Load MSMARCO (BEIR) ####
#############################
dataset = "msmarco"
# url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
# out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
# data_path = util.download_and_unzip(url, out_dir)

data_path = "/home/ukp/thakur/projects/sbert_retriever/datasets-new/msmarco"

### Load BEIR MSMARCO training dataset, this will be used for query and corpus for reference.
corpus, queries, qrels = GenericDataLoader(data_path).load(split="train")
corpus = {doc_id: corpus[doc_id]["title"] + corpus[doc_id]["text"] for doc_id in corpus}

train_queries = {}

for qid in qrels:
    train_queries[qid] = {'query': queries[qid],
                            'pos': set(qrels[qid]),
                            'soft-pos': set(),
                            'neg': set()}

logging.info("Clean train queries")
deleted_queries = 0
for qid in list(train_queries.keys()):
    if len(train_queries[qid]['pos']) == 0:
        deleted_queries += 1
        del train_queries[qid]
        continue

logging.info("Deleted queries pos-empty: {}".format(deleted_queries))

##################################################
#### Download MSMARCO Hard Negs Triplets File ####
##################################################

triplets_url = "https://sbert.net/datasets/msmarco-hard-negatives.jsonl.gz"
msmarco_triplets_filepath = os.path.join(data_path, "msmarco-hard-negatives.jsonl.gz")
if not os.path.isfile(msmarco_triplets_filepath):
    util.download_url(triplets_url, msmarco_triplets_filepath)

with gzip.open(msmarco_triplets_filepath, 'rt') as fIn:
    for line in tqdm.tqdm(fIn, total=502939):
        data = json.loads(line)
        qid = data['qid']

        if qid in train_queries:
            neg_added = 0
            max_neg_added = 100
            
            for keyword in data['neg']:
                if "dot-score" in data['neg'][keyword][0]:
                    key = "dot-score"
                elif "bm25-score" in data['neg'][keyword][0]:
                    key = "bm25-score"
                else:
                    key = "cos-score"

                hits = sorted(data['neg'][keyword], key=lambda x: x[key], reverse=True)
                
                for hit in hits:
                    pid = hit['corpus_id'] if 'corpus_id' in hit else hit['pid']

                    if pid in train_queries[qid]['pos']:    #Skip entries we have as positives
                        continue

                    if hit['ce-score'] < 0.1 and neg_added < max_neg_added:
                        train_queries[qid]['neg'].add(pid)
                        neg_added += 1
                    elif hit['ce-score'] > 0.9:
                        train_queries[qid]['soft-pos'].add(pid)
        
        if len(train_queries) > 100:
            break

logging.info("Clean train queries with empty neg set")
deleted_queries = 0
for qid in list(train_queries.keys()):
    if len(train_queries[qid]['neg']) == 0:
        deleted_queries += 1
        del train_queries[qid]
        continue

logging.info("Deleted queries neg empty: {}".format(deleted_queries))
train_queries = list(train_queries.values())
for idx in range(len(train_queries)):
    train_queries[idx]['pos'] = list(train_queries[idx]['pos'])
    train_queries[idx]['neg'] = list(train_queries[idx]['neg'])
    train_queries[idx]['soft-pos'] = list(train_queries[idx]['soft-pos'])


##########################################
#### Getting Model Ready for Training ####
##########################################
# Prepare optimizers
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = transformers.AdamW(model.parameters(), lr=lr, eps=1e-6)   #optimizer_grouped_parameters
t_total = math.ceil(len(train_queries)/batch_size*num_epochs)
num_warmup_steps = int(t_total/grad_acc_steps * 0.1)    #10% for warm up

# Scheduler
scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)

# Scaler
scaler = torch.cuda.amp.GradScaler()

# Loss function: Cross Entropy Loss
loss_fct = torch.nn.CrossEntropyLoss()

# Train the model
for epoch in tqdm.trange(num_epochs, desc='Epochs'):
    random.shuffle(train_queries)
    idx = 0
    for start_idx in tqdm.trange(0, len(train_queries), batch_size):
        idx += 1
        if idx > 5000 and idx % 5000 == 0:
            model.bert_model.save_pretrained(model_save_path)
            logging.info(f"Save to {model_save_path}")

        batch = train_queries[start_idx:start_idx+batch_size]
        queries = [b['query'] for b in batch]

        #First the positives
        passages = [corpus[random.choice(b['pos'])] for b in batch]

        #Then the negatives
        for b in batch:
            for pid in random.sample(b['neg'], k=min(len(b['neg']), num_negatives)):
                passages.append(corpus[pid])

        label = torch.tensor(list(range(len(batch))), device=device)

        ##FP16
        with autocast():
            final_scores = model(queries, passages) 
            final_scores = 10*final_scores
            loss_value = loss_fct(final_scores, label) / grad_acc_steps

        scaler.scale(loss_value).backward()
        if (idx + 1) % grad_acc_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            model.zero_grad()
            scheduler.step()

# Save the final model
model.bert_model.save_pretrained(model_save_path)
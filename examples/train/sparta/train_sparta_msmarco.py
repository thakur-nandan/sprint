from torch.utils.data import DataLoader
from sentence_transformers import losses, util, models
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer, evaluation
from sentence_transformers.readers import InputExample
import logging
from datetime import datetime
import os
from shutil import copyfile
import sys
import math
import gzip
import random
import tqdm
from transformers import AutoTokenizer, AutoModel, BertModel
import transformers
import torch
from SPARTA import SPARTA
import json
import numpy as np
from torch.cuda.amp import autocast
import os
from shutil import copyfile
import datetime
from collections import defaultdict
from scipy.sparse import csc_matrix, csr_matrix

random.seed(42)

scaler = torch.cuda.amp.GradScaler()

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# Fill GPU

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_name = sys.argv[1]
model = SPARTA(model_name, device)

model_save_path = "output/msmarco-{}-{}".format(model_name.rstrip("/").split("/")[-1], datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
model.tokenizer.save_pretrained(model_save_path)


##Distil setting
if 'distil' in model_name:
    batch_size, num_negatives = 4, 35
else:
    batch_size, num_negatives = 3, 20

logging.info(f"batch_size: {batch_size}")
logging.info(f"num_neg: {num_negatives}")


# Write self to path
os.makedirs(model_save_path, exist_ok=True)

train_script_path = os.path.join(model_save_path, 'train_script.py')
copyfile(__file__, train_script_path)
with open(train_script_path, 'a') as fOut:
    fOut.write("\n\n# Script was called via:\n#python " + " ".join(sys.argv))


########################
corpus = {}
train_queries = {}





#### Read dev file
logging.info("Create dev dataset")
dev_corpus_max_size = 100*1000

dev_queries_file = 'data/queries.dev.small.tsv'
needed_pids = set()
needed_qids = set()
dev_qids = set()

dev_queries = {}
dev_corpus = {}
dev_rel_docs = {}

with open(dev_queries_file) as fIn:
    for line in fIn:
        qid, query = line.strip().split("\t")
        dev_qids.add(qid)

with open('data/qrels.dev.tsv') as fIn:
    for line in fIn:
        qid, _, pid, _ = line.strip().split('\t')

        if qid not in dev_qids:
            continue

        if qid not in dev_rel_docs:
            dev_rel_docs[qid] = set()
        dev_rel_docs[qid].add(pid)

        needed_pids.add(pid)
        needed_qids.add(qid)

with open(dev_queries_file) as fIn:
    for line in fIn:
        qid, query = line.strip().split("\t")
        if qid in needed_qids:
            dev_queries[qid] = query

with gzip.open('data/collection-rnd.tsv.gz', 'rt') as fIn:
    for line in fIn:
        pid, passage = line.strip().split("\t")
        if pid in needed_pids or dev_corpus_max_size <= 0 or len(dev_corpus) <= dev_corpus_max_size:
            dev_corpus[pid] = passage

dev_corpus_pids = list(dev_corpus.keys())
dev_corpus = [dev_corpus[pid] for pid in dev_corpus_pids]

########### Eval functions

def compute_passage_emb(passages):
    sparse_embeddings = []
    bert_input_emb = model.bert_model.embeddings.word_embeddings(torch.tensor(list(range(0, len(model.tokenizer))), device=device))
    sparse_vec_size = 2000

    # Set Special tokens [CLS] [MASK] etc. to zero
    for special_id in model.tokenizer.all_special_ids:
        bert_input_emb[special_id] = 0 * bert_input_emb[special_id]

    with torch.no_grad():
        tokens = model.tokenizer(passages, padding=True, truncation=True, return_tensors='pt', max_length=500).to(device)
        passage_embeddings = model.bert_model(**tokens).last_hidden_state
        for passage_emb in passage_embeddings:
            scores = torch.matmul(bert_input_emb, passage_emb.transpose(0, 1))
            max_scores = torch.max(scores, dim=-1).values
            relu_scores = torch.relu(max_scores) #Eq. 5
            final_scores = torch.log(relu_scores + 1)  # Eq. 6, final score

            top_results = torch.topk(final_scores, k=sparse_vec_size, sorted=True)
            passage_emb = defaultdict(float)
            for score, idx in zip(top_results[0].cpu().tolist(), top_results[1].cpu().tolist()):
                if score > 0:
                    passage_emb[idx] = score
                else:
                    break

            sparse_embeddings.append(passage_emb)

    return sparse_embeddings

def evaluate_msmarco():
    passage_embs_sorted = []
    batch_size = 32

    length_sorted_idx = np.argsort([-len(pas) for pas in dev_corpus])
    dev_corpus_sorted = [dev_corpus[idx] for idx in length_sorted_idx]

    for start_idx in tqdm.trange(0, len(dev_corpus_sorted), batch_size, desc='encode corpus'):
        passage_embs_sorted.extend(compute_passage_emb(dev_corpus_sorted[start_idx:start_idx + batch_size]))

    passage_embs = [passage_embs_sorted[idx] for idx in np.argsort(length_sorted_idx)]

    logging.info("Create sparse matrix")
    row = []
    col = []
    values = []
    for pid, emb in enumerate(passage_embs):
        for tid, score in emb.items():
            row.append(tid)
            col.append(pid)
            values.append(score)

    sparse = csr_matrix((values, (row, col)), shape=(len(model.tokenizer), len(passage_embs)), dtype=np.float)
    logging.info("Scores: {}".format(sparse.shape))

    mrr = []
    k = 10
    for qid, question in tqdm.tqdm(dev_queries.items(), desc="score"):
        token_ids = model.tokenizer(question, add_special_tokens=False)['input_ids']

        # Get the candidate passages
        scores = np.asarray(sparse[token_ids, :].sum(axis=0)).squeeze(0)
        top_k_ind = np.argpartition(scores, -k)[-k:]
        hits = sorted([(dev_corpus_pids[pid], scores[pid]) for pid in top_k_ind], key=lambda x: x[1], reverse=True)

        mrr_score = 0
        for rank, hit in enumerate(hits[0:10]):
            pid = hit[0]
            if pid in dev_rel_docs[qid]:
                mrr_score = 1 / (rank + 1)
                break
        mrr.append(mrr_score)

    assert len(mrr) == len(dev_queries)
    mrr = np.mean(mrr)
    logging.info("MRR@10: {:.4f}".format(mrr))
    return mrr


best_score = 0 #evaluate_msmarco()

#################


#### Read train file

with gzip.open('data/collection.tsv.gz', 'rt') as fIn:
    for line in fIn:
        pid, passage = line.strip().split("\t")
        corpus[pid] = passage


with open('data/queries.train.tsv', 'r') as fIn:
    for line in fIn:
        qid, query = line.strip().split("\t")
        train_queries[qid] = {'query': query,
                              'pos': set(),
                              'soft-pos': set(),
                              'neg': set()}



#Read qrels file for relevant positives per query
with open('data/qrels.train.tsv') as fIn:
    for line in fIn:
        qid, _, pid, _ = line.strip().split()
        train_queries[qid]['pos'].add(pid)


logging.info("Clean train queries")
deleted_queries = 0
for qid in list(train_queries.keys()):
    if len(train_queries[qid]['pos']) == 0:
        deleted_queries += 1
        del train_queries[qid]
        continue

logging.info("Deleted queries pos-empty: {}".format(deleted_queries))

for hard_neg_file in ['data/hard-negatives-all.jsonl.gz']: #'data/hard-negatives-ann-roberta.jsonl.gz']: #['data/hard-negatives-ann-msmarco-distilbert-base-v2.jsonl.gz', 'data/hard-negatives-ann.jsonl.gz', 'data/hard-negatives-ann-no_idnt.jsonl.gz', 'data/hard-negatives-all.jsonl.gz']:
    logging.info("Read hard negatives: "+hard_neg_file)
    with gzip.open(hard_neg_file, 'rt') as fIn:
        try:
            for line in fIn:
                try:
                    data = json.loads(line)
                except:
                    continue
                qid = data['qid']

                if qid in train_queries:
                    neg_added = 0
                    max_neg_added = 100

                    hits = sorted(data['hits'], key=lambda x: x['score'] if 'score' in x else x['bm25-score'], reverse=True)
                    for hit in hits:
                        pid = hit['corpus_id'] if 'corpus_id' in hit else hit['pid']

                        if pid in train_queries[qid]['pos']:    #Skip entries we have as positives
                            continue

                        if hit['bert-score'] < 0.1 and neg_added < max_neg_added:
                            train_queries[qid]['neg'].add(pid)
                            neg_added += 1
                        elif hit['bert-score'] > 0.9:
                            train_queries[qid]['soft-pos'].add(pid)
        except:
            pass


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



###########################################




####
# Prepare optimizers
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]



grad_acc_steps, lr = 1, 2e-5
#grad_acc_steps, lr = 16, 2e-5


num_epochs = 1
optimizer = transformers.AdamW(model.parameters(), lr=lr, eps=1e-6)   #optimizer_grouped_parameters
t_total = math.ceil(len(train_queries)/batch_size*num_epochs)
num_warmup_steps = int(t_total/grad_acc_steps * 0.1)    #10% for warm up
scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)
loss_fct = torch.nn.CrossEntropyLoss()
max_grad_norm = 1


for epoch in tqdm.trange(num_epochs, desc='Epochs'):
    random.shuffle(train_queries)
    idx = 0
    for start_idx in tqdm.trange(0, len(train_queries), batch_size):
        idx += 1
        if (idx) % 5000 == 0:
            score = evaluate_msmarco()
            if score > best_score:
                best_score = score
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


        """
        #Normal FP32 with grad acc
        final_scores = model(query, passages)
        #Compute loss
        loss_value = loss_fct(final_scores, label)
        if grad_acc_steps > 1:
            loss_value /= grad_acc_steps
        loss_value.backward()
    
        if (idx+1) % grad_acc_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            model.zero_grad()
            scheduler.step()
        """


logging.info("Final eval:")
evaluate_msmarco()
from torch.utils.data import Dataset
from beir import util
import os
import linecache
import json

class BeIRDataIterator(Dataset):

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.nexamples = len(linecache.getlines(data_dir))
    
    def __getitem__(self, index):
        if index >= self.nexamples:
            raise StopIteration

        line = linecache.getline(self.data_dir, index + 1)
        line_dict = json.loads(line)
        example = {}
        example['id'] = line_dict['_id']
        example['text'] = ' '.join([line_dict['title'], line_dict['text']])
        return example

    def __len__(self):
        return self.nexamples


def beir(dataset, data_dir=None):
    if data_dir is None:
        assert dataset is not None
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
        out_dir = 'datasets/beir'
        data_dir = os.path.join(out_dir, dataset)
        if not os.path.exists(data_dir):
            util.download_and_unzip(url, out_dir)

    return BeIRDataIterator(os.path.join(data_dir, 'corpus.jsonl'))


def build(data_name, data_dir):
    data_name = data_name.lower()
    
    if 'beir' in data_name:
        data_name = data_name.replace('beir/', '')
        data_name = data_name.replace('beir_', '')
        return beir(data_name, data_dir)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    data_iter = build('beir/scifact', None)
    print(len(data_iter))
    for i, example in enumerate(data_iter):
        if i > 3:
            break
        print(example)
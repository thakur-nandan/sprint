import csv
import json
import os
import platform
import re
import subprocess
from typing import Dict, List

import numpy as np

def load_qrels(qrels_path) -> Dict[str, Dict[str, float]]:
    # adapted from BeIR: 
    # https://github.com/beir-cellar/beir/blob/main/beir/datasets/data_loader.py#L114
    reader = csv.reader(open(qrels_path, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    next(reader)  # skip header
    
    qrels = {}
    for row in reader:
        query_id, corpus_id, score = row[0], row[1], int(row[2])
        
        if query_id not in qrels:
            qrels[query_id] = {corpus_id: score}
        else:
            qrels[query_id][corpus_id] = score
    
    return qrels


def load_queries(queries_path) -> Dict[str, str]:
    # adapted from BEIR:
    # https://github.com/beir-cellar/beir/blob/main/beir/datasets/data_loader.py#L107
    queries = {}
    with open(queries_path, encoding='utf8') as fIn:
        for line in fIn:
            line = json.loads(line)
            queries[line.get("_id")] = line.get("text")
    
    return queries


def get_processor_name() -> str:
    """Reference: https://stackoverflow.com/a/13078519/16409125."""
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Darwin":
        os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
        command ="sysctl -n machdep.cpu.brand_string"
        return subprocess.check_output(command).strip()
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).decode().strip()
        for line in all_info.split("\n"):
            if "model name" in line:
                return re.sub(".*model name.*:", "", line,1)
    return "Cannot get processor info."


def get_folder_size(start_path: str) -> str:
    """Reference: https://stackoverflow.com/a/1392549/16409125."""
    total_size = 0
    for dirpath, _, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return f"{round(total_size / 1024 / 1024, 2)}MB"


def bin_and_average(keys: List[float], values: List[float], numpy_bins: List[int]) -> List[float]:
    """Reference: https://stackoverflow.com/a/6163403/16409125."""
    digitized = np.digitize(keys, numpy_bins)
    values = np.array(values)
    bin_means = [values[digitized == i].mean() for i in range(1, len(numpy_bins))]
    return bin_means


def bin_and_std(keys: List[float], values: List[float], numpy_bins: List[int]) -> List[float]:
    """Reference: https://stackoverflow.com/a/6163403/16409125."""
    digitized = np.digitize(keys, numpy_bins)
    values = np.array(values)
    bin_stds = [values[digitized == i].std() for i in range(1, len(numpy_bins))]
    return bin_stds

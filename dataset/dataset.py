import fileinput
import json
import os
import pandas as pd
from .utils import normalize_logs, cat2codes, set_global_seeds, question_difficult, skill_difficult
import random
from collections import defaultdict
import numpy as np
import torch
from functools import reduce
from collections import defaultdict
from .gkt_utils import get_gkt_graph, get_skt_graph

def load_data(dataset, argument=False, multi_concept=False):
    path = f'data/processed/{dataset}'
    train_data, valid_data, test_data = [], [], []
    if not argument:
        with open(f'{path}/train.json', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                train_data.append(json.loads(line))
        with open(f'{path}/valid.json', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                valid_data.append(json.loads(line))
        with open(f'{path}/test.json', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                test_data.append(json.loads(line))
        with open(f'{path}/info.json', 'r', encoding='utf-8') as f:
            info = json.load(f)
        with open(f'{path}/know_map.json', 'r', encoding='utf-8') as f:
            know_map = json.load(f)
        with open(f'{path}/id2concept.json', 'r', encoding='utf-8') as f:
            id2concept = json.load(f)
        return train_data, valid_data, test_data, info, know_map, id2concept

def load_csv_data(filepath_df, seed=None, train_size=None, valid_size=None, info: dict = None, rename=None, min_length=3,
                 truncate=None, max_length=200, shuffle=True, drop_columns=[], encoding='utf-8', use_it=False, use_at=False, max_interval_time=None, max_taken_time=None, 
                 usecols: list = None, renamecols: dict = None, dropna=None, astype=None, nrows=None,transform_func=None,
                 **kwargs):
    set_global_seeds(seed)
    data = []
    info = {}
    if astype is None:
        astype = {
            "user": "category",
            "item": "category",
            "concept": "category",
            "correct": "uint8"
        }
    if dropna is None:
        dropna = ["concept"]
    if not isinstance(filepath_df, str):
        df = filepath_df.rename(columns=renamecols).dropna(
            subset=dropna
        ).astype(astype
        ).drop(columns=drop_columns)
    elif filepath_df.split('.')[-1]=='csv':
        df = pd.read_csv(
            filepath_df,
            nrows=nrows,
            usecols=usecols,
            encoding=encoding
        ).rename(columns=renamecols).dropna(
            subset=dropna
        ).astype(astype
        ).drop(columns=drop_columns)
    elif filepath_df.split('.')[-1]=='txt':
        df = pd.read_table(
            filepath_df,
            nrows=nrows,
            usecols=usecols,
            encoding=encoding
        ).rename(columns=renamecols).dropna(
            subset=dropna
        ).astype(astype
        ).drop(columns=drop_columns)
    else:
        raise ValueError()
    
    if transform_func:
        df = transform_func(df)
    print('load raw data successfully')
    it_set = set()
    if use_at:
        if not max_taken_time:
            df['taken_time'] = df['taken_time'].astype('category')
        info['taken_time_num'] = len(df.taken_time.unique())
    else:
        df['taken_time'] = 0
        df['taken_time'] = df['taken_time'].astype('category')
        info['taken_time_num'] = 0
    id2concept = {i+1:c for i,c in enumerate(list(df.concept.values.categories))}
    df = cat2codes(df)
    df['item'] = df['item'] + 1
    df['concept'] = df['concept'] + 1
    df = df.reset_index(drop=True)

    know_map = defaultdict(list)
    for s, p in zip(df.concept.tolist(), df.item.tolist()):
        know_map[p].append(s)
    for p,s in know_map.items():
        know_map[p] = list(set(s))
    info['record_num'] = df.shape[0]
    info['item_num'] = len(df.item.unique())
    info['concept_num'] = len(df.concept.unique())
    info['raw_seq_num'] = len(df.user.unique())
    info['raw_avg_seq_len'] = df.groupby("user").count()['item'].mean()

    log_idx = 0
    processed_seq_len_list = []
    for user, user_seq in df.groupby("user"):
        if len(user_seq) < min_length:
            continue
        logs = user_seq.sort_values(by="time_index").reset_index(drop=True)
        if use_it:
            logs['time_index'] = (logs['time_index'] - logs['time_index'][0])
            for i in range(1, len(logs['time_index'])):
                item = (logs['time_index'][i] - logs['time_index'][i-1])//60
                if item > max_interval_time:
                    item = max_interval_time
                it_set.add(item)
                
        logs = normalize_logs(logs, truncate, max_length)
        for _logs in logs:
            _logs = _logs.to_dict("list")
            _logs["user"] = [log_idx] * len(_logs["item"])
            log_idx += 1
            data.append(_logs)
            processed_seq_len_list.append(len(_logs["item"]))
            # yield cls.split_log(_logs, train_size, seed=seed)
    if use_it:
        it2id = { a: i for i, a in enumerate(it_set) }
        info['interval_time_num'] = len(it2id)
        for seq in data:
            it = [0]
            for i in range(1, len(seq['time_index'])):
                item = (seq['time_index'][i] - seq['time_index'][i - 1])//60
                if item > max_interval_time:
                    item = max_interval_time
                it.append(it2id[item])
            seq['interval_time'] = it
    else:
        info['interval_time_num']= 0
        for seq in data:
            seq['interval_time'] = [0]*len(seq['item'])

    info['processed_avg_seq_len'] = np.array(processed_seq_len_list).mean()
    info['processed_seq_num'] = len(data)
    if shuffle:
        random.shuffle(data)
    boundary1 = round(len(data) * (train_size+valid_size))
    
    df_train_valid = defaultdict(list)
    for logs in data[: boundary1]:
        for key in logs:
            df_train_valid[key].extend(logs[key])
    df_train_valid = dict(df_train_valid)
    df_train_valid = pd.DataFrame(df_train_valid)
    sd = skill_difficult(df_train_valid)
    qd = question_difficult(df_train_valid)
    for logs in data:
        logs['qd'] = [qd[x] if x in qd else 0 for x in logs['item']]
        logs['sd'] = [sd[x] if x in sd else 0 for x in logs['concept']]
        historyratios = []
        right, total = 0, 0
        for i in range(0, len(logs['concept'])):
            if logs['correct'][i] == 1:
                right += 1
            total += 1
            historyratios.append(right / total)
        logs['historycorrs'] = historyratios
    
    train_data, test_data = data[: boundary1], data[boundary1:]
    boundary2 = round(len(data) * train_size)
    train_data, valid_data = train_data[:boundary2], train_data[boundary2:]
    graph, graph_json = get_gkt_graph(info['concept_num']+1, data[: boundary1])
    correct_graph_json, ctrans_graph_json = get_skt_graph(info['concept_num']+1, data[: boundary1])
    
    return train_data, valid_data, test_data, info, know_map, qd, sd, id2concept, graph, graph_json, correct_graph_json, ctrans_graph_json



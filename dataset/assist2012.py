from .dataset import load_csv_data
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import json
import os
import pandas as pd
import numpy as np

def assist2012(filepath, seed=0, train_size=0.7, valid_size=0.1, batch_size=128, encoding='utf-8', save=True
               , save_argument=False, merge_multi_concept=False,multi_concept=False):
    print('preprocessing assist2012')
    def transform_func(df):
        df['time_index'] = pd.to_datetime(df['time_index']).astype('int')//1e9
        df['taken_time'] = df['taken_time'] // 1000
        return df
    train_data, valid_data, test_data, info, know_map, qd, sd, id2concept, graph, graph_json, correct_graph_json, ctrans_graph_json = load_csv_data(
        filepath,
        usecols=[
            "skill",
            "problem_id",
            "user_id",
            "start_time",
            "correct",
            "overlap_time"
        ],
        renamecols={
            "user_id": "user",
            "problem_id": "item",
            "skill": "concept",
            "correct": "correct",
            "start_time": "time_index",
            "overlap_time": "taken_time"
        },
        dropna=["concept"],
        astype={
            "user": "category",
            "item": "category",
            "concept": "category",
            "correct": "uint8",
            "taken_time": "int"
        },
        seed=seed,
        train_size = train_size,
        valid_size=valid_size,
        encoding=encoding,
        use_it=True,
        use_at=True,
        transform_func = transform_func,
        max_interval_time=43200,
        max_taken_time=None,
        max_length=100,
        merge_multi_concept = merge_multi_concept
    )
    print('save processed data')
    keys = ['item', 'concept', 'correct', 'interval_time', 'taken_time', 'time_index', 'qd', 'sd', 'historycorrs']
    path = 'data/processed/assist2012'
    if merge_multi_concept:
        path += '/merge'
    if save:
        with open(f'{path}/train.json', 'w+', encoding='utf-8') as f:
            for seq in train_data:
                f.write(json.dumps({key: seq[key] for key in keys})+'\n')
        with open(f'{path}/valid.json', 'w+', encoding='utf-8') as f:
            for seq in valid_data:
                f.write(json.dumps({key: seq[key] for key in keys})+'\n')
        with open(f'{path}/test.json', 'w+', encoding='utf-8') as f:
            for seq in test_data:
                f.write(json.dumps({key: seq[key] for key in keys})+'\n')
        with open(f'{path}/test.txt', 'w+', encoding='utf-8') as f:
            for i, seq in enumerate(test_data):
                f.write(str(i) + ',' + str(len(seq['item'])) + '\n')
                f.write(','.join([str(s) for s in seq['item']]) + '\n')
                f.write(','.join([str(a) for a in seq['concept']]) + '\n')
                f.write(','.join([str(p) for p in seq['correct']]) + '\n')
                f.write(','.join([str(i) for i in seq['taken_time']]) + '\n')
                f.write(','.join([str(a) for a in seq['interval_time']]) + '\n')
                f.write(','.join([str(a) for a in seq['time_index']]) + '\n')
        with open(f'{path}/info.json', 'w+', encoding='utf-8') as f:
            json.dump(info, f, indent=4)
        with open(f'{path}/qd.json', 'w+', encoding='utf-8') as f:
            json.dump(qd, f, indent=4)
        with open(f'{path}/sd.json', 'w+', encoding='utf-8') as f:
            json.dump(sd, f, indent=4)
        with open(f'{path}/know_map.json', 'w+', encoding='utf-8') as f:
            json.dump(know_map, f, indent=4)
        with open(f'{path}/id2concept.json', 'w+', encoding='utf-8') as f:
            json.dump(id2concept, f, indent=4)
        with open(f'{path}/transition_graph.json', 'w+', encoding='utf-8') as f:
            json.dump(graph_json, f, indent=4)
        with open(f'{path}/correct_transition_graph.json', 'w+', encoding='utf-8') as f:
            json.dump(correct_graph_json, f, indent=4)
        with open(f'{path}/ctrans_sim_graph.json', 'w+', encoding='utf-8') as f:
            json.dump(ctrans_graph_json, f, indent=4)
        np.savez(f'{path}/transition_graph.npz', matrix = graph)
    if save_argument:
        with open(f'{path}/train_argument.json', 'w+', encoding='utf-8') as f:
            for seq in train_data:
                for i in range(3, len(seq['item'])+1):
                    f.write(json.dumps({'item': seq['item'][:i], 'concept': seq['concept'][:i], 'correct': seq['correct'][:i]})+'\n')
        with open(f'{path}/valid_argument.json', 'w+', encoding='utf-8') as f:
            for seq in valid_data:
                for i in range(3, len(seq['item'])+1):
                    f.write(json.dumps({'item': seq['item'][:i], 'concept': seq['concept'][:i], 'correct': seq['correct'][:i]})+'\n')
        with open(f'{path}/test_argument.json', 'w+', encoding='utf-8') as f:
            for seq in test_data:
                for i in range(3, len(seq['item'])+1):
                    f.write(json.dumps({'item': seq['item'][:i], 'concept': seq['concept'][:i], 'correct': seq['correct'][:i]})+'\n')
        with open(f'{path}/info.json', 'w+', encoding='utf-8') as f:
            json.dump(info, f, indent=4)
        with open(f'{path}/know_map.json', 'w+', encoding='utf-8') as f:
            json.dump(know_map, f, indent=4)
    return train_data, valid_data, test_data, info, know_map
import pandas as pd
from typing import Dict, List
import math
import numpy as np
import random
import torch
from datetime import datetime
from tqdm import tqdm
def set_global_seeds(i):
    np.random.seed(i)
    random.seed(i)
    torch.manual_seed(i)
    torch.cuda.manual_seed(i)
    torch.cuda.manual_seed_all(i)

def cat2codes(df: pd.DataFrame) -> pd.DataFrame:
    for column in df.select_dtypes(['category']).columns:
        df[column] = df[column].cat.codes
    return df

def normalize_logs(logs, truncate=False, max_length=None) -> List:
    if truncate is True:
        assert max_length is not None
        logs = [logs[:max_length]]
    elif truncate is None and max_length is not None:
        logs = [logs[i: i + max_length] for i in range(0, len(logs), max_length)]
    else:
        logs = [logs]
    return logs

def skill_difficult(df,concepts='concept',responses='correct',diff_level=100):
    sd = {}
    # df = df.reset_index(drop=True)
    set_skills = set(np.array(df[concepts]).tolist())
    for i in tqdm(set_skills):
        count = 0
        idx = df[(df[concepts] == i)].index.tolist()
        tmp_data = df.iloc[idx]
        correct_1 = tmp_data[responses]
        if len(idx) < 30:
            sd[i] = 0
            continue
        else:
            for j in np.array(correct_1):
                count += j
            if count == 0:
                sd[i] = 1
                continue
            else:
                avg = int((count/len(correct_1))*diff_level)+1
                sd[i] = avg

    return sd

def question_difficult(df,questions='item',responses='correct',diff_level=100):
    qd = {}
    # df = df.reset_index(drop=True)
    set_questions = set(np.array(df[questions]).tolist())
    for i in tqdm(set_questions):
        count = 0
        idx = df[(df[questions] == i)].index.tolist()
        tmp_data = df.iloc[idx]
        correct_1 = tmp_data[responses]
        if len(idx) < 30:
            qd[i] = 0
            continue
        else:
            for j in np.array(correct_1):
                count += j
            if count == 0:
                qd[i] = 1
                continue
            else:
                avg = int((count/len(correct_1))*diff_level)+1
                qd[i] = avg

    return qd
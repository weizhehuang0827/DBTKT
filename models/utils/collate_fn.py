import torch
from torch.nn.utils.rnn import pad_sequence
from functools import reduce
import numpy as np

def seq_time2_collate_fn(batch):
    q = [torch.tensor(sample['item'], dtype=torch.long) for sample in batch]
    c = [torch.tensor(sample['concept'], dtype=torch.long) for sample in batch]
    r = [torch.tensor(sample['correct'], dtype=torch.long) for sample in batch]
    at = [torch.tensor(sample['taken_time'], dtype=torch.long) for sample in batch]
    it = [torch.tensor(sample['interval_time'], dtype=torch.long) for sample in batch]
    ti = [torch.tensor(sample['time_index'], dtype=torch.long) for sample in batch]
    q = pad_sequence(q, batch_first=True, padding_value=-1)
    c = pad_sequence(c, batch_first=True, padding_value=-1)
    r = pad_sequence(r, batch_first=True, padding_value=-1)
    at = pad_sequence(at, batch_first=True, padding_value=0)
    it = pad_sequence(it, batch_first=True, padding_value=0)
    ti = pad_sequence(ti, batch_first=True, padding_value=0)
    return q,c,r,at,it,ti

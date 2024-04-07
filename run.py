import fire
import os
import torch
import torch.nn as nn
from sklearn import metrics
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import logging
from dataset import load_data
from dataset.utils import set_global_seeds
from models import DBTKT
from models.utils import seq_time2_collate_fn
import glob
from datetime import datetime, timedelta

def run(seed=0, dataset='assist2017', model_name='dkt', desc='', lr=0.001, epochs=100, device='cuda', batch_size=64, num_workers=4, ques_window=2, concept_window=2, from_file='',
        gpu_no=0, l2=0.0002, output_dir='output', early_stop=5, use_qid=True,use_time=None, use_at=None,use_it=None,n=1,lr_decay_step=10,lr_decay_rate=0.5, dropout=0.2, d_k=128):

    if device == 'cuda':
        device = device + f':{gpu_no}'
    set_global_seeds(seed)

    cur_time = datetime.now()
    cur_time = cur_time.strftime("%Y%m%d_%H%M%S")
    # logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler("./logs/"+"_".join([model_name,dataset,cur_time,desc])+'.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    
    print('loading data')
    train_data, valid_data, test_data, info, know_map, id2concept = load_data(dataset, argument=False)
    
    
    
    if use_at is None:
        use_at = (dataset not in ['nips2020', 'aaai2023'])
    if use_it is None:
        use_it = True
    if model_name in ['dbtkt']:
        collate_fn = seq_time2_collate_fn
    kwargs = dict(dataset=dataset, model_name=model_name, lr=lr, batch_size=batch_size,epochs=epochs, 
                  dropout=dropout, d_k=d_k,device=device,gpu_no=gpu_no, desc=desc,use_at=use_at,use_it=use_it,
                  seed=seed)
    logger.info(kwargs)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        # num_workers=num_workers,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=batch_size,
        shuffle=False,
        # num_workers=num_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        # num_workers=num_workers,
        collate_fn=collate_fn,
    )

    if model_name == 'dbtkt':
        model = DBTKT(n_at=info['taken_time_num']+2, n_it=info['interval_time_num']+2, 
                        n_exercise=info['item_num']+2, n_question=info['concept_num']+2, d_a=50, d_e=d_k, d_k=d_k, dropout=0.2, use_at=use_at,use_it=use_it)
    if from_file:
        model.load_state_dict(torch.load(from_file, map_location=lambda s, _: s))
    optim = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=l2
    )
    # scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_decay_step, gamma=lr_decay_rate)
    model.to(device)
    criterion = nn.BCELoss()
    
    print('training')
    logger.info('training')
    #training
    best = {"auc": 0}
    best_epoch = 0
    best_model_name = ''
    for epoch in range(epochs):
        total_loss = 0.0
        total_cnt = 0
        if model_name not in ['iekt','qikt']:
            model.train()
        else:
            model.model.train()
        train_it = tqdm(train_loader, desc=f'EPOCH {epoch}')
        for batch in train_it:
            if model_name in ['dbtkt']:
                q, c, r, at, it, ti = batch
                q,r,at,it,c,ti = q.to(device), r.to(device), at.to(device), it.to(device), c.to(device), ti.to(device)
                loss = model.get_loss(criterion, q,at,r,it,c,ti)
            optim.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            optim.step()
            total_loss += loss.item()
            total_cnt += 1
            postfix = {"loss": total_loss / total_cnt}
            train_it.set_postfix(postfix)
            
        # scheduler.step()
        
        if model_name not in ['iekt','qikt']:
            model.eval()
        else:
            model.model.eval()
        with torch.no_grad():
            pred_list = []
            true_list = []
            for batch in tqdm(valid_loader, desc=f'EPOCH {epoch}'):
                if model_name in ['dbtkt']:
                    q, c, r, at, it, ti = batch
                    q,r,at,it,c,ti = q.to(device), r.to(device), at.to(device), it.to(device), c.to(device), ti.to(device)
                    masked_pred, masked_truth = model.get_eval(q,at,r,it,c,ti)
                pred_list.extend(masked_pred.cpu().tolist())
                true_list.extend(masked_truth.cpu().tolist())
                
            r = {
                "acc": metrics.accuracy_score(true_list, np.asarray(pred_list).round()),
                "auc": metrics.roc_auc_score(true_list, pred_list),
                "mae": metrics.mean_absolute_error(true_list, pred_list),
                "rmse": metrics.mean_squared_error(true_list, pred_list) ** 0.5,
            }
            print(f"valid report: {model_name}-{dataset}-{desc}-{epoch:03d}-{r['auc']:.4f}")
            logger.info(f"valid report: {model_name}-{dataset}-{desc}-{epoch:03d}-{r['auc']:.4f}")
            if r["auc"] > best["auc"]:
                delete_path = model_path = os.path.join(
                        output_dir, f"{model_name}-{dataset}-{desc}-*-{cur_time}"
                    )
                for i in glob.glob(delete_path):
                    os.remove(i)
                best = r
                best_epoch = epoch
                best_model_name = f"{model_name}-{dataset}-{desc}-{epoch:03d}-{r['auc']:.4f}-{cur_time}.pt"
                if output_dir:
                    model_path = os.path.join(
                        output_dir, best_model_name
                    )
                    print("saving snapshot to:", model_path)
                    logger.info(f"saving snapshot to: {model_path}")
                    torch.save(model.state_dict(), model_path)

            if early_stop > 0 and epoch - best_epoch > early_stop:
                print(f"did not improve for {early_stop} epochs, stop early")
                break
    print("best epoch:", best_epoch)
    print("best result", {k: f"{v:.4f}" for k, v in best.items()})
    
    print('testing')
    
    logger.info("best epoch:"+ str(best_epoch))
    logger.info("best result"+ str({k: f"{v:.4f}" for k, v in best.items()}))
    
    logger.info('testing')
    #testing
    if output_dir:
        best_model_path = os.path.join(
                        output_dir, best_model_name
                    )
        model.load_state_dict(torch.load(best_model_path, map_location=lambda s, _: s))
    model.to(device)
    if model_name not in ['iekt','qikt']:
        model.eval()
    else:
        model.model.eval()
    with torch.no_grad():
        pred_list = []
        true_list = []
        for batch in tqdm(test_loader, desc=f'EPOCH {epoch}'):
            if model_name in ['dbtkt']:
                q, c, r, at, it, ti = batch
                q,r,at,it,c,ti = q.to(device), r.to(device), at.to(device), it.to(device), c.to(device), ti.to(device)
                masked_pred, masked_truth = model.get_eval(q,at,r,it,c,ti)
            pred_list.extend(masked_pred.cpu().tolist())
            true_list.extend(masked_truth.cpu().tolist())
        r = {
            "acc": metrics.accuracy_score(true_list, np.asarray(pred_list).round()),
            "auc": metrics.roc_auc_score(true_list, pred_list),
            "mae": metrics.mean_absolute_error(true_list, pred_list),
            "rmse": metrics.mean_squared_error(true_list, pred_list) ** 0.5,
        }
        print(f"test report: {model_name}-{dataset}-{desc}-{r['auc']:.4f}")
        logger.info(f"test report: {model_name}-{dataset}-{desc}-auc-{r['auc']:.4f}")
        logger.info(f"test report: {model_name}-{dataset}-{desc}-acc-{r['acc']:.4f}")
        logger.info(f"test report: {model_name}-{dataset}-{desc}-rmse-{r['rmse']:.4f}")




if __name__ == '__main__':
    fire.Fire(run)
    # run(dataset='assist2017',model_name='transkt',batch_size=32,lr=0.001,gpu_no=1)
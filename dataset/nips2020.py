import os
import pandas as pd
from datetime import datetime
from .dataset import load_csv_data
import json
import numpy as np
# from .utils import sta_infos, write_txt, change2timestamp,format_list2str
def change2timestamp(t, hasf=True):
    if hasf:
        # timeStamp = datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f").timestamp() * 1000
        timeStamp = datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f").timestamp()
    else:
        # timeStamp = datetime.strptime(t, "%Y-%m-%d %H:%M:%S").timestamp() * 1000
        timeStamp = datetime.strptime(t, "%Y-%m-%d %H:%M:%S").timestamp()
    return int(timeStamp)
def format_list2str(input_list):
    return [str(x) for x in input_list]
def sta_infos(df, keys, stares, split_str="_"):
    # keys: 0: uid , 1: concept, 2: question
    uids = df[keys[0]].unique()
    if len(keys) == 2:
        cids = df[keys[1]].unique()
    elif len(keys) > 2:
        qids = df[keys[2]].unique()
        ctotal = 0
        cq = df.drop_duplicates([keys[2], keys[1]])[[keys[2], keys[1]]]
        cq[keys[1]] = cq[keys[1]].fillna("NANA")
        cids, dq2c = set(), dict()
        for i, row in cq.iterrows():
            q = row[keys[2]]
            ks = row[keys[1]]
            dq2c.setdefault(q, set())
            if ks == "NANA":
                continue
            for k in str(ks).split(split_str):
                dq2c[q].add(k)
                cids.add(k)
        ctotal, na, qtotal = 0, 0, 0
        for q in dq2c:
            if len(dq2c[q]) == 0:
                na += 1 # questions has no concept
                continue
            qtotal += 1
            ctotal += len(dq2c[q])
        
        avgcq = round(ctotal / qtotal, 4)
    avgins = round(df.shape[0] / len(uids), 4)
    ins, us, qs, cs = df.shape[0], len(uids), "NA", len(cids)
    avgcqf, naf = "NA", "NA"
    if len(keys) > 2:
        qs, avgcqf, naf = len(qids), avgcq, na
    curr = [ins, us, qs, cs, avgins, avgcqf, naf]
    stares.append(",".join([str(s) for s in curr]))
    return ins, us, qs, cs, avgins, avgcqf, naf

def write_txt(file, data):
    with open(file, "w") as f:
        for dd in data:
            for d in dd:
                f.write(",".join(d) + "\n")
def load_nips_data(primary_data_path,meta_data_dir,task_name):
    """The data downloaded from https://competitions.codalab.org/competitions/25449 
    The document can be downloaded from https://arxiv.org/abs/2007.12061.

    Args:
        primary_data_path (_type_): premary data path
        meta_data_dir (_type_): metadata dir
        task_name (_type_): task_1_2 or task_3_4

    Returns:
        dataframe: the merge df
    """
    print("Start load data")
    answer_metadata_path = os.path.join(meta_data_dir,f"answer_metadata_{task_name}.csv")
    question_metadata_path = os.path.join(meta_data_dir,f"question_metadata_{task_name}.csv")
    student_metadata_path = os.path.join(meta_data_dir,f"student_metadata_{task_name}.csv")
    subject_metadata_path = os.path.join(meta_data_dir,f"subject_metadata.csv")
    
    df_primary = pd.read_csv(primary_data_path)
    print(f"len df_primary is {len(df_primary)}")
    #add timestamp
    df_answer = pd.read_csv(answer_metadata_path)
    df_answer['answer_timestamp'] = df_answer['DateAnswered'].apply(change2timestamp)
    df_question = pd.read_csv(question_metadata_path)
    # df_student = pd.read_csv(student_metadata_path)
    df_subject = pd.read_csv(subject_metadata_path)
    
    #only keep level 3
    keep_subject_ids = set(df_subject[df_subject['Level']==3]['SubjectId'])
    df_question['SubjectId_level3'] = df_question['SubjectId'].apply(lambda x:set(eval(x))&keep_subject_ids)
    
    
    #merge data
    df_merge = df_primary.merge(df_answer[['AnswerId','answer_timestamp']],how='left')#merge answer time
    df_merge = df_merge.merge(df_question[["QuestionId","SubjectId_level3"]],how='left')#merge question subjects
    df_merge['SubjectId_level3_str'] = df_merge['SubjectId_level3'].apply(lambda x:"_".join([str(i) for i in x]))
    print(f"len df_merge is {len(df_merge)}")
    print("Finish load data")
    print(f"Num of student {df_merge['UserId'].unique().size}")
    print(f"Num of question {df_merge['QuestionId'].unique().size}")
    kcs =[]
    for item in df_merge['SubjectId_level3'].values:
        kcs.extend(item)
    print(f"Num of knowledge {len(set(kcs))}")
    print(len(df_merge['SubjectId_level3_str'].unique()))
    return df_merge

def get_user_inters(df):
    """convert df to user sequences 

    Args:
        df (_type_): the merged df

    Returns:
        List: user_inters
    """
    user_inters = []
    for user, group in df.groupby("UserId", sort=False):
        group = group.sort_values(["answer_timestamp","tmp_index"], ascending=True)

        seq_skills = group['SubjectId_level3_str'].tolist()
        seq_ans = group['IsCorrect'].tolist()
        seq_response_cost = ["NA"]
        seq_start_time = group['answer_timestamp'].tolist()
        seq_problems = group['QuestionId'].tolist()
        seq_len = len(group)
        user_inters.append(
            [[str(user), str(seq_len)],
             format_list2str(seq_problems),
             format_list2str(seq_skills),
             format_list2str(seq_ans),
             format_list2str(seq_start_time),
             format_list2str(seq_response_cost)])
    return user_inters


KEYS = ["UserId", "SubjectId_level3_str", "QuestionId"]
    
def nips2020(primary_data_path,meta_data_dir,task_name,write_file,seed=0, train_size=0.7, valid_size=0.1, batch_size=128, encoding='utf-8', save=True
               , save_argument=False, merge_multi_concept=False,multi_concept=False):
    stares= []
    df = load_nips_data(primary_data_path,meta_data_dir,task_name)
    
    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(f"original interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")
    
    df['tmp_index'] = range(len(df))
    df = df.dropna(subset=["UserId","answer_timestamp", "SubjectId_level3_str", "IsCorrect", "answer_timestamp","QuestionId"])
    
    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(f"after drop interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")

    user_inters = get_user_inters(df)
    write_txt(write_file, user_inters)
    
    print('preprocessing nips2020')
    train_data, valid_data, test_data, info, know_map, qd, sd, id2concept, graph, graph_json,correct_graph_json, ctrans_graph_json = load_csv_data(
        df,
        usecols = ["UserId","answer_timestamp", "SubjectId_level3_str", "IsCorrect", "answer_timestamp","QuestionId"],
        renamecols={
            "UserId": "user",
            "QuestionId": "item",
            "answer_timestamp": "time_index",
            "SubjectId_level3_str": "concept",
            "IsCorrect": "correct",
        },
        dropna=["concept", "item"],
        astype={
                "user": "category",
                "item": "category",
                "concept": "category",
                "correct": "uint8",
            },
        seed=seed,
        train_size = train_size,
        valid_size=valid_size,
        encoding=encoding,
        use_at=False,
        use_it=True,
        max_length=100,
        max_interval_time=43200,
        merge_multi_concept = merge_multi_concept
    )
    subject_metadata_path = os.path.join(meta_data_dir,f"subject_metadata.csv")
    df_subject = pd.read_csv(subject_metadata_path)
    subjectid2name = {}
    for _,row in df_subject.iterrows():
        subjectid2name[str(row['SubjectId'])] = row['Name']
    for id, concept in id2concept.items():
        concept = [subjectid2name[str(c)] for c in concept.split('_')]
        concept = '_'.join(concept)
        id2concept[id] = concept
    print('save processed data')
    keys = keys = ['item', 'concept', 'correct', 'interval_time', 'taken_time', 'time_index', 'qd', 'sd','historycorrs']
    path = 'data/processed/nips2020'
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

if __name__ == '__main__':
    nips2020('/data/huangweizhe/KT/data/NIPS2020/public_data/train_data/train_task_3_4.csv', '/data/huangweizhe/KT/data/NIPS2020/public_data/metadata', "task_3_4", '/data/huangweizhe/KT/data/processed/nips2020result.txt')
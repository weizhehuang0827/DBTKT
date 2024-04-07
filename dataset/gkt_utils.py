import torch
import json
import os
import numpy as np
import pandas as pd

def get_gkt_graph(num_c, data, graph_type="transition"):
    graph = None
    # df_train = pd.read_csv(os.path.join(dpath, trainfile))
    # df_test = pd.read_csv(os.path.join(dpath, testfile))
    # df = pd.concat([df_train, df_test])  
    graph_json = []
    if graph_type == 'dense':
        graph = build_dense_graph(num_c)
        graph = graph.numpy()
    elif graph_type == 'transition':
        graph = build_transition_graph(data, num_c)
        graph = graph.numpy()
    index_i, index_j = np.nonzero(graph)
    index_i, index_j = index_i.tolist(), index_j.tolist()
    graph_list = graph.tolist()
    for i,j in zip(index_i,index_j):
        graph_json.append([i,j,graph_list[i][j]])
    # with open(f'{path}/transition_graph.json', 'w+', encoding='utf-8') as f:
    #     json.dump(graph_json, f, indent=4)
    # np.savez(f'{path}/transition_graph.npz', matrix = graph)
    return graph, graph_json

def get_skt_graph(num_c, data):
    correct_graph_json = []
    ctrans_graph_json = []
    
    correct_graph, ctrans_graph = build_correct_ctrans_graph(data, num_c)
    correct_graph = correct_graph.numpy()
    index_i, index_j = np.nonzero(correct_graph)
    index_i, index_j = index_i.tolist(), index_j.tolist()
    correct_graph_list = correct_graph.tolist()
    for i,j in zip(index_i,index_j):
        correct_graph_json.append([i,j,correct_graph_list[i][j]])
    ctrans_graph = ctrans_graph.numpy()
    index_i, index_j = np.nonzero(ctrans_graph)
    index_i, index_j = index_i.tolist(), index_j.tolist()
    ctrans_graph_list = ctrans_graph.tolist()
    for i,j in zip(index_i,index_j):
        ctrans_graph_json.append([i,j,ctrans_graph_list[i][j]])
    
    return correct_graph_json, ctrans_graph_json
    
    
def build_transition_graph(data, concept_num):
    """generate transition graph

    Args:
        df (da): _description_
        concept_num (int): number of concepts

    Returns:
        numpy: graph
    """
    graph = np.zeros((concept_num, concept_num))
    for seq in data:
        questions = seq['concept']
        for i in range(len(questions)-1):
            pre = int(questions[i])
            next = int(questions[i+1])
            graph[pre, next] += 1
    np.fill_diagonal(graph, 0)
    # row normalization
    rowsum = np.array(graph.sum(1))

    def inv(x):
        if x == 0:
            return x
        return 1. / x

    inv_func = np.vectorize(inv)
    r_inv = inv_func(rowsum).flatten()
    r_mat_inv = np.diag(r_inv)
    graph = r_mat_inv.dot(graph)
    graph = torch.from_numpy(graph).float()
    
    return graph

def build_correct_ctrans_graph(data, concept_num):
    graph = np.zeros((concept_num, concept_num))
    for seq in data:
        questions = seq['concept']
        correct = seq['correct']
        for i in range(len(questions)-1):
            pre = int(questions[i])
            next = int(questions[i+1])
            if correct[i]==1 and correct[i+1]==1:
                graph[pre, next] += 1
    np.fill_diagonal(graph, 0)
    # row normalization
    rowsum = np.array(graph.sum(1))

    def inv(x):
        if x == 0:
            return x
        return 1. / x

    inv_func = np.vectorize(inv)
    r_inv = inv_func(rowsum).flatten()
    r_mat_inv = np.diag(r_inv)
    graph = r_mat_inv.dot(graph)
    graph = torch.from_numpy(graph).float()
    ctrans_graph = (graph+graph.T)/(torch.abs(graph-graph.T)+0.1)
    ctrans_graph = (ctrans_graph - torch.min(ctrans_graph))/(torch.max(ctrans_graph)-torch.min(ctrans_graph))
    return graph,ctrans_graph

def build_dense_graph(concept_num):
    """generate dense graph

    Args:
        concept_num (int): number of concepts

    Returns:
        numpy: graph
    """
    graph = 1. / (concept_num - 1) * np.ones((concept_num, concept_num))
    np.fill_diagonal(graph, 0)
    graph = torch.from_numpy(graph).float()
    return graph
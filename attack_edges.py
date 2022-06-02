from __future__ import division
from __future__ import print_function

import time
import os
import networkx as nx
import matplotlib.pyplot as plt

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
import numpy as np
import scipy.sparse as sp

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from gae.optimizer import OptimizerAE, OptimizerVAE
from gae.input_data_club import load_data
from gae.model import GCNModelAE, GCNModelVAE
from gae.preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges
from igraph import *
import igraph
from sklearn import metrics
from community_detection import community_detection
import ast





def get_sorted_edges(g,datasets,topa,topd,community,G):

    with open("results/logs/{}_sigmoid_adj.txt".format(datasets),'r') as fa:
        sigmoid_adj = np.loadtxt(fa, dtype=np.float32, delimiter=' ')


    copyComm = []
    for k in community:
        k = list(k)
        k.sort()
        copyComm.append(k)
    copyComm.sort()

    community = []
    for k in copyComm:
        community.append(set(k))


    low1 = 
    low2 = 
    high1 = 
    high2 = 

    rows = sigmoid_adj.shape[0]
    da = {}
    dd = {}
    for k in range(len(community)):
        for i in range(rows):
            for j in range(i):
                if low1 < sigmoid_adj[i][j] < low2:
                    if (i in community[k] and j not in community[k]) or (i not in community[k] and j in community[k]):
                        da[(i, j)] = round(sigmoid_adj[i][j], )
                        # print(i,j,"=",sigmoid_adj[i][j])
                if high1 < sigmoid_adj[i][j] < high2:
                    if (i in community[k] and j in community[k]) or (j in community[k] and i in community[k]):
                        dd[(i, j)] = sigmoid_adj[i][j]
    for k, v in dd.items():
        dd[k] -= 
        dd[k] = round(dd[k], )

    sumsda = {}
    sumsda.update(da)
    sumsda.update(dd)

    tempsum = {sumsda[key]: key for key in sumsda}
    sumsda = {tempsum[key]: key for key in tempsum}

    sumsda = sorted(sumsda.items(), key=lambda item: abs(item[1]))




    topa = topa
    topd = topd
    sda = {}
    sdd = {}
    sd_ =[]

    for i in sumsda[:topa]:
        sda[i[0]] = i[1]

    print("sda: ",sda)
    community = community

    cntd = 0
    cnta = 0

    Ka = []
    Kd = []
    A = []
    for i in community:
        A.append(i)
    for i in range(len(community)):
        for j in A[i]:
            G.nodes[j]["commA"] = i


    tempi = -1
    tempj = -1
    for k,v in sda.items():
        if v > 0:
            # print(k)
            i, j = k

            if (G.nodes[i]["commA"] != G.nodes[j]["commA"]):



                Ka.append(k)

                cnta += 1
                tempi = i
                tempj = j
        else:
        
            i, j = k
            if (i == tempi) or (j == tempj):
                continue

            if (G.nodes[i]["commA"] == G.nodes[j]["commA"]):
                try:
                    
                    Kd.append(k)
                    
                    cntd += 1
                    tempi = i
                    tempj = j
              
                except igraph._igraph.InternalError:
                    continue
        if (cnta + cntd) >= topd:
            break
    print("add edges: ", cnta)
    print("add edges list: ",Ka)




    print("del edges: ",cntd)
    print("del edges list: ",Kd)
    print("budgets: ",cnta+cntd)
    return Ka, Kd


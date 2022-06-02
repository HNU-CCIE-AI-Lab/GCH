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
import sys
# f = open('print.log', 'a')
# sys.stdout = f


# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', , 'Initial learning rate.')
flags.DEFINE_integer('epochs', , 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', , 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', , 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', , 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', , 'Dropout rate (1 - keep probability).')
flags.DEFINE_string('model', 'gcn_ae', 'Model string.')
flags.DEFINE_string('dataset', 'miserables', 'Dataset string.')
flags.DEFINE_integer('features', , 'Whether to use features (1) or not (0).')

model_str = FLAGS.model
dataset_str = FLAGS.dataset

# Load data
adj, features = load_data(dataset_str)


# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
adj = adj_train

if FLAGS.features == 0:
    features = sp.identity(features.shape[0])  # featureless

# Some preprocessing
adj_norm = preprocess_graph(adj)

# Define placeholders
placeholders = {
    'features': tf.sparse_placeholder(tf.float32),
    'adj': tf.sparse_placeholder(tf.float32),
    'adj_orig': tf.sparse_placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=())
}

num_nodes = adj.shape[0]

features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
features_nonzero = features[1].shape[0]

# Create model
model = None
if model_str == 'gcn_ae':
    model = GCNModelAE(placeholders, num_features, features_nonzero)
elif model_str == 'gcn_vae':
    model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)

pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

# Optimizer
with tf.name_scope('optimizer'):
    if model_str == 'gcn_ae':
        opt = OptimizerAE(preds=model.reconstructions,
                          labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                      validate_indices=False), [-1]),
                          pos_weight=pos_weight,
                          norm=norm)
    elif model_str == 'gcn_vae':
        opt = OptimizerVAE(preds=model.reconstructions,
                           labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                       validate_indices=False), [-1]),
                           model=model, num_nodes=num_nodes,
                           pos_weight=pos_weight,
                           norm=norm)

# Initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_val = []
acc_val = []


def get_roc_score(edges_pos, edges_neg, emb=None):
    if emb is None:
        feed_dict.update({placeholders['dropout']: 0})
        emb = sess.run(model.z_mean, feed_dict=feed_dict)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    shuffle=[]


    adj_rec = np.dot(emb, emb.T)
    # adj_rec = tf.nn.sigmoid(np.dot(emb, emb.T))
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    #print(tf.nn.sigmoid(adj_rec))
    sigmoid_adj = sigmoid(adj_rec)


    sig_emb = sigmoid(emb)
    #print(np.around(sigmoid_adj))


    return roc_score, ap_score, sigmoid_adj, emb, sig_emb


cost_val = []
acc_val = []
val_roc_score = []

adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = sparse_to_tuple(adj_label)

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    # Run single weight update
    outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)

    # Compute average loss
    avg_cost = outs[1]
    avg_accuracy = outs[2]

    roc_curr, ap_curr, sigmoid_adj, emb, sig_emb = get_roc_score(val_edges, val_edges_false)

    val_roc_score.append(roc_curr)

    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
          "train_acc=", "{:.5f}".format(avg_accuracy), "val_roc=", "{:.5f}".format(val_roc_score[-1]),
          "val_ap=", "{:.5f}".format(ap_curr),
          "time=", "{:.5f}".format(time.time() - t))
    Adj = np.around(sigmoid_adj).astype(int) - np.eye(adj.shape[0])

    acc = 0


    if avg_accuracy >= acc :
        break
topa = 

print("Optimization Finished!")


roc_score, ap_score, sigmoid_adj, emb, sig_emb = get_roc_score(test_edges, test_edges_false)
print('Test ROC score: ' + str(roc_score))
print('Test AP score: ' + str(ap_score))

Adj = np.around(sigmoid_adj).astype(int) - np.eye(adj.shape[0])


G = nx.from_numpy_matrix(Adj)
g = nx.from_scipy_sparse_matrix(adj)



rows = adj.shape[0]

da = {}
dd = {}


sigmoid_adj = sigmoid_adj - np.diag(np.diag(sigmoid_adj))
np.savetxt('{}_sigmoid_adj.txt'.format(FLAGS.dataset),sigmoid_adj)



g = nx.from_scipy_sparse_matrix(adj)
adj = nx.to_numpy_array(g)
g = Graph.Adjacency(adj.astype(bool).tolist(),mode='undirected')

community = g.community_multilevel()
communitys = g.community_multilevel()

summary(g)

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

for k in range(len(community)):
    for i in range(rows):
        for j in range(i):
            if low1 < sigmoid_adj[i][j] < low2:
                if (i in community[k] and j not in community[k]) or (i not in community[k] and j in community[k]):
                    da[(i,j)]=round(sigmoid_adj[i][j],6)
                    # print(i,j,"=",sigmoid_adj[i][j])
            if high1 < sigmoid_adj[i][j] < high2:
                if (i in community[k] and j in community[k]) or (j in community[k] and i in community[k]):
                    dd[(i,j)] = sigmoid_adj[i][j]
for k,v in dd.items():
    dd[k] -= 
    dd[k] = round(dd[k],0)

sumsda = {}
sumsda.update(da)
sumsda.update(dd)

tempsum = {sumsda[key]:key for key in sumsda}
sumsda = {tempsum[key]:key for key in tempsum}

sumsda = sorted(sumsda.items(), key = lambda item:abs(item[1]))
filesa = open('{}_sumsda.txt'.format(FLAGS.dataset),'w')
filesa.write(str(sumsda))
filesa.close()

print("sumsda: ",sumsda)




topd = 
sda = {}
sdd = {}

for i in sumsda[:topa]:
    sda[i[0]] = i[1]


print("sda: ",sda)






cntd = 0
cnta = 0
Kd = []
Ka = []
for c in range(len(community)):
    for k,v in sda.items():

        # print(


        if v <= 0:
            i, j = k
            if (i in community[c] and j in community[c]) or (j in community[c] and i in community[c]):
                try:
                    g.delete_edges(g.get_eid(i,j))
                    Kd.append(k)
                    print("delete edge: ",k)
                    cntd += 1
                # except nx.exception.NetworkXError:
                except igraph._igraph.InternalError:
                    continue
        else:
            tempi = -1
            tempj = -1

            # print(k)
            i, j = k
            if (i == tempi) or (j == tempj):
                continue
            if (i in community[c] and j not in community[c]):
                g.add_edge(*k)
                Ka.append(k)
                print("add edge: ", k)
                cnta += 1
                tempi = i
                tempj = j

print("del edges list: ",Kd)


print("add edges list: ",Ka)




comm = g.community_multilevel()




print("del edges: ",cntd)
print("add edges：",cnta)
for c in range(len(community)):
    print("原始网络社区%d元素个数："%c,len(community[c]))
print(communitys)
for c in range(len(comm)):
    print("攻击后网络社区%d元素个数："%c,len(comm[c]))
print(comm)


copyComm = []
for k in comm:
    k = list(k)
    k.sort()
    copyComm.append(k)
copyComm.sort()

comm = []
for k in copyComm:
    comm.append(set(k))






for c in range(len(comm)):

    nout = len(community[c] - comm[c]) / len(community[c])
    nin = len(comm[c] - community[c]) / len(community[c])
    score = nout + nin




    print("The number of %d community out nodes: "%c,len(community[c] - comm[c]),": ",community[c] - comm[c],nout)
    print("The number of %d community in nodes: "%c,len(comm[c] - community[c]),": ",comm[c] - community[c],nin)

    print("Total score: ",score)

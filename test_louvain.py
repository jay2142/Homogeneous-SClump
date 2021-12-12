from pysclump import PathSim
import numpy as np
import networkx as nx
from pysclump import *
import argparse
from sklearn.cluster import SpectralClustering
from sklearn.metrics.cluster import rand_score
from tqdm import tqdm
import community
import csv
import os 
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def save_graph(graph,file_name, cluster_assignments):
    #initialze Figure
    plt.figure(num=None, figsize=(20, 20), dpi=80)
    plt.axis('off')
    fig = plt.figure(1)
    pos = nx.spring_layout(graph)
    # labels = nx.get_edge_attributes(G, 'label')
    # cmap = cm.get_cmap('viridis', max(labels.values()))
    nx.draw_networkx_nodes(graph,pos,cmap=cm.get_cmap('viridis',max(cluster_assignments)), node_color=cluster_assignments)
    nx.draw_networkx_edges(graph,pos)
    # nx.draw_networkx_labels(graph,pos)

    cut = 1.00
    xmax = cut * max(xx for xx, yy in pos.values())
    ymax = cut * max(yy for xx, yy in pos.values())
    plt.xlim(-xmax, xmax)
    plt.ylim(-ymax, ymax)

    plt.savefig(file_name,bbox_inches="tight")
    del fig

a = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
a.add_argument("--data", help="name of dataset directory",
               required=True)
a.add_argument("--het-source", help="source of heterogeneity", 
               choices=["degree", "clustering"], default="clustering")
a.add_argument("--num-clusterings", help="number of pre-clusterings to average", 
               type=int, default=3)
args = a.parse_args()

N_NODES = 6301

adjacency = np.zeros((N_NODES, N_NODES))

# load edges and target clusters
print("loading edges and target clusters...")
with open('data/p2p-Gnutella08.txt') as f:
    csv_reader = csv.reader(f, delimiter='\t')
    for i in range(4):
        next(csv_reader)
    for a, b in csv_reader:
        a = int(a)
        b = int(b)
        adjacency[a, b] = 1
adjacency = np.maximum(adjacency, adjacency.T)

print(adjacency)
# construct graph
G = nx.convert_matrix.from_numpy_matrix(adjacency)



# cluster
partition = community.best_partition(G)
cluster_list = []
i = 0
while i in partition.keys():
    cluster_list.append(partition[i])
    i+=1

i = 0
# make cluster nodes alphebetic to prevent clash with integer valued nodes
cluster_assignments = np.asarray([str(i)+chr(x+65) for x in cluster_list])
print(cluster_assignments)

# save_graph(G, "test_louvain.pdf", cluster_assignments)

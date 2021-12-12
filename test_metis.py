from pysclump import PathSim
import numpy as np
import networkx as nx
from pysclump import *
import argparse
from sklearn.cluster import SpectralClustering
from sklearn.metrics.cluster import rand_score
from tqdm import tqdm
import metis
import csv
import os 


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
edgecuts, parts = metis.part_graph(G, 15)
cluster_list = parts

i = 0
# make cluster nodes alphebetic to prevent clash with integer valued nodes
cluster_assignments = np.asarray([str(i)+chr(x+65) for x in cluster_list])
print(cluster_assignments)

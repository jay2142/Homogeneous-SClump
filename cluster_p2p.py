from pysclump import PathSim
import numpy as np
import networkx as nx
from pysclump import *
import argparse
from sklearn.cluster import SpectralClustering
from sklearn.metrics.cluster import rand_score
from tqdm import tqdm
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
k = [15,17,20]

clustering_labels = []
for i in range(args.num_clusterings):
    # cluster
    clustering = SpectralClustering(n_clusters=k[i], affinity='precomputed', random_state=i)
    clustering.fit(adjacency)

    # make cluster nodes alphebetic to prevent clash with integer valued nodes
    cluster_assignments = np.asarray([str(i)+chr(x+65) for x in clustering.labels_.tolist()])

    # add a new node for each cluster
    for cluster in np.unique(cluster_assignments):
        G.add_node(cluster)

    # connect new nodes to their associated clusters
    for node in G.nodes:
        if isinstance(node, int):
            G.add_edge(node, cluster_assignments[node])
            
    # save clustering labels for rand score later
    clustering_labels.append(clustering.labels_)


# check that clusterings are unique...
print("rand score between preclusterings:", rand_score(clustering_labels[0], clustering_labels[1]))

# recompute adjacency matrix
A = nx.adjacency_matrix(G).toarray()

# group 
group = np.full(N_NODES, 'A')
group = np.concatenate(([group] + [np.full(k[i], chr(i+66)) for i in range(args.num_clusterings)]))

# create type lists
print("creating type lists...")
type_lists = {}
for type in np.unique(group):
    type_lists[type] = np.where(group==type)[0].tolist()

# create incidence matrices
print("constructing incidence matrices...")
if args.het_source == "degree": 
    # TODO: de-hardcode or make args
    links = ['AB']
elif args.het_source == "clustering":
    links = ['A'+cluster for cluster in np.unique(group)]

incidence_matrices = {}
for link in links:
    incidence_matrices[link] = A[np.ix_(type_lists[link[0]], type_lists[link[1]])]

# Create PathSim instance.
print("creating PathSim instance...")
ps = PathSim(type_lists, incidence_matrices)

# Construct similarity matrices.
print("constructing similarity matrices...")
if args.het_source == "degree":
    # TODO: de-hardcode these metapaths or make them args
    clustered_type = 'B'
    metapaths = ['BAB']
elif args.het_source == "clustering":
    metapaths = ['A'+cluster+'A' for cluster in np.delete(np.unique(group),0)]
print(metapaths)

similarity_matrices = {}
for metapath in tqdm(metapaths):
    similarity_matrices[metapath] = PathSim.compute_similarity_matrix(ps, metapath=metapath)

# Create SClump instance.
print("creating SClump instance...")
sclump = SClump(similarity_matrices, num_clusters=17)

# Run the algorithm!
print("running SClump...")
labels, learned_similarity_matrix, metapath_weights = sclump.run(verbose=True)

# print results
print(labels, learned_similarity_matrix, metapath_weights)
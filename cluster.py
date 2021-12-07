from pysclump import PathSim
import numpy as np
import networkx as nx
from sclump import *
import argparse
from sklearn.cluster import SpectralClustering
from sklearn.metrics.cluster import rand_score
from tqdm import tqdm
import os 

a = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
a.add_argument("--het-source", help="source of heterogeneity", choices=["degree", "clustering"])
a.add_argument("--data", help="name of dataset directory")
a.add_argument("--num_clusterings", help="number of pre-clusterings to average")
args = a.parse_args()

# load edges and target clusters
print("loading edges and target clusters...")
path = os.getcwd()
edges_path = os.path.join(path, "data", args.data, "edges.csv")
edges = np.loadtxt(open(edges_path, "rb"), delimiter=",", skiprows=1).astype('int')
target_path = os.path.join(path, "data", args.data, "target.csv")
target_clusters = np.loadtxt(open(target_path, "rb"), delimiter=",", skiprows=1).astype('int')

k = np.max(target_clusters[:,1]) + 1
n_nodes = np.unique(edges).shape[0]

# construct graph
print("constructing graph...")
G = nx.Graph()
G.add_nodes_from(range(n_nodes))
G.add_edges_from(edges)
A = nx.adjacency_matrix(G).toarray()

## heterogenize graph
print(f"heterogenizing by {args.het_source}...")

if args.het_source == "degree":
    group = []
    degrees = np.sum(A, axis=1)

    # TODO: de-hardcode or make args
    for i in range(n_nodes):
        if degrees[i] <= 1: group.append('A')
        elif degrees[i] <= 2: group.append('B')
        else: group.append('C')

    group = np.asarray(group)

elif args.het_source == "clustering":
    # cluster
    print("\tclustering...")
    clustering = SpectralClustering(n_clusters=k, affinity='precomputed')
    clustering.fit(A)

    print("\trecomputing adjacency matrix...")
    # make groups alphebetic to prevent clash with integer valued nodes
    group = np.asarray([chr(x+65) for x in clustering.labels_.tolist()])

    # add a new node for each cluster
    for cluster in np.unique(group):
        G.add_node(cluster)

    # connect new nodes to their associated clusters
    for node in G.nodes:
        if isinstance(node, int):
            G.add_edge(node, group[node])

    # recompute adjacency matrix
    A = nx.adjacency_matrix(G).toarray()

    # redo group array such that all original nodes are one group, 
    # and all new nodes are their own group
    original_node_type = chr(np.unique(group).shape[0]+65)
    group = np.concatenate((np.full(n_nodes, original_node_type), np.unique(group)))

# create type lists
print("creating type lists...")
type_lists = {}
for type in np.unique(group):
    type_lists[type] = np.where(group==type)[0].tolist()

# create incidence matrices
print("creating incidence matrices...")
if args.het_source == "degree": 
    # TODO: de-hardcode or make args
    links = ['AB']
elif args.het_source == "clustering":
    links = [original_node_type + cluster for cluster in np.unique(group)]

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
    metapaths = [original_node_type + cluster + original_node_type for cluster in np.unique(group)]

similarity_matrices = {}
for metapath in tqdm(metapaths):
    similarity_matrices[metapath] = PathSim.compute_similarity_matrix(ps, metapath=metapath)

# Create SClump instance.
print("creating SClump instance...")
sclump = SClump(similarity_matrices, num_clusters=k)

# Run the algorithm!
print("running SClump...")
labels, learned_similarity_matrix, metapath_weights = sclump.run(verbose=True)

# print results
print("metapath weights:", metapath_weights)
if args.het_source == "degree":
    print("rand score:", rand_score(labels, target_clusters[type_lists[clustered_type],1]))
    print(labels)
elif args.het_source == "clustering": 
    print("rand score, preclustering:", rand_score(clustering.labels_, target_clusters[:,1]))
    print("rand score, sclump:", rand_score(labels, target_clusters[:,1]))
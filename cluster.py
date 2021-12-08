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
a.add_argument("--data", help="name of dataset directory",
               required=True)
a.add_argument("--het-source", help="source of heterogeneity", 
               choices=["degree", "clustering"], default="clustering")
a.add_argument("--num-clusterings", help="number of pre-clusterings to average", 
               type=int, default=2)
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
    clustering_labels = []
    for i in tqdm(range(args.num_clusterings)):
        # cluster
        clustering = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=i)
        clustering.fit(A)

        # make cluster nodes alphebetic to prevent clash with integer valued nodes
        cluter_assignments = np.asarray([str(i)+chr(x+65) for x in clustering.labels_.tolist()])

        # add a new node for each cluster
        for cluster in np.unique(cluter_assignments):
            G.add_node(cluster)

        # connect new nodes to their associated clusters
        for node in G.nodes:
            if isinstance(node, int):
                G.add_edge(node, cluter_assignments[node])
                
        # save clustering labels for rand score later
        clustering_labels.append(clustering.labels_)

    # check that clusterings are unique...
    print("rand score between preclusterings:", rand_score(clustering_labels[0], clustering_labels[1]))

    # recompute adjacency matrix
    A = nx.adjacency_matrix(G).toarray()

    # group 
    group = np.full(n_nodes, 'A')
    group = np.concatenate(([group] + [np.full(k, chr(i+66)) for i in range(args.num_clusterings)]))

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
    print("rand scores of preclusterings:", 
          [rand_score(clustering_labels[i], target_clusters[:,1]) for i in range(args.num_clusterings)])
    print("rand score, sclump:", rand_score(labels, target_clusters[:,1]))
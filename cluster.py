from pysclump import PathSim
import numpy as np
import networkx as nx
from pysclump import *
import argparse
from sklearn.cluster import SpectralClustering
from sklearn.metrics.cluster import rand_score
from tqdm import tqdm
import community
import metis
import os 
import random
import copy

VARY_K = True
VARY_ALGORITHM = False

def run_louvain(G):
    partition = community.best_partition(G)
    cluster_list = []
    i = 0
    while i in partition.keys():
        cluster_list.append(partition[i])
        i+=1
    print(len(cluster_list))
    return cluster_list

def run_metis(G):
    _, cluster_list = metis.part_graph(G, 18)
    print(len(cluster_list))
    return cluster_list

def run_spectral_clustering(G):
    A = nx.adjacency_matrix(G).toarray()
    clustering = SpectralClustering(n_clusters=k, affinity='precomputed', 
                                        random_state=random.randint(0,1e9))
    clustering.fit(A)
    print(len(clustering.labels_.tolist()))
    return clustering.labels_.tolist()
    


a = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
a.add_argument("--data", help="name of dataset directory",
               required=True)
a.add_argument("--het-source", help="source of heterogeneity", 
               choices=["degree", "clustering"], default="clustering")
a.add_argument("--num-clusterings", help="number of pre-clusterings to average", 
               type=int, default=3)
args = a.parse_args()

# load edges and target clusters
print("loading edges and target clusters...")
path = os.getcwd()
edges_path = os.path.join(path, "data", args.data, "edges.csv")
edges = np.loadtxt(open(edges_path, "rb"), delimiter=",", skiprows=1).astype('int')
target_path = os.path.join(path, "data", args.data, "target.csv")
target_clusters = np.loadtxt(open(target_path, "rb"), delimiter=",", skiprows=1).astype('int')

target_k = np.max(target_clusters[:,1]) + 1
n_nodes = np.unique(edges).shape[0]

# construct graph
print("constructing graph...")
G = nx.Graph()
G.add_nodes_from(range(n_nodes))
G.add_edges_from(edges)

G_augmented = copy.deepcopy(G)

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
    clustering_methods = [run_louvain, run_metis, run_spectral_clustering]
    clustering_labels = []
    group = np.full(n_nodes, 'A')
    for i in tqdm(range(args.num_clusterings)):
        if VARY_K:
            k = (target_k//args.num_clusterings)*(i+1)
        else: 
            k = target_k
        
        if not VARY_ALGORITHM:
            labels = run_spectral_clustering(G)
        else:
            labels = clustering_methods[i](G)

        # make cluster nodes alphebetic to prevent clash with integer valued nodes
        cluster_assignments = np.asarray([str(i)+chr(x+65) for x in labels])

        print(cluster_assignments)

        # add a new node for each cluster
        for cluster in np.unique(cluster_assignments):
            G_augmented.add_node(cluster)

        # connect new nodes to their associated clusters
        for node in G.nodes:
            if isinstance(node, int):
                G_augmented.add_edge(node, cluster_assignments[node])
                
        # save clustering labels for rand score later
        clustering_labels.append(labels)
        
        # update group matrix
        group = np.concatenate(([group] + [np.full(k, chr(i+66))]))

    # check that clusterings are unique
    rand_scores = np.zeros((args.num_clusterings, args.num_clusterings))
    for i in range(args.num_clusterings):
        for j in range(i+1,args.num_clusterings):
            rand_scores[i][j] = rand_score(clustering_labels[i], clustering_labels[j])
    print("rand scores between preclusterings:\n", rand_scores)

    # recompute adjacency matrix
    A = nx.adjacency_matrix(G_augmented).toarray()
    print(A.shape)

# create type lists
print("creating type lists...")
type_lists = {}
for type in np.unique(group):
    type_lists[type] = np.where(group==type)[0].tolist()
# print(type_lists)

# create incidence matrices
print("constructing incidence matrices...")
if args.het_source == "degree": 
    # TODO: de-hardcode or make args
    links = ['AB']
elif args.het_source == "clustering":
    links = ['A'+cluster for cluster in np.unique(group)]

# print(A.shape)

incidence_matrices = {}
for link in links:
    # print(link)
    # print(np.ix_(type_lists[link[0]], type_lists[link[1]]))
    incidence_matrices[link] = A[np.ix_(type_lists[link[0]], type_lists[link[1]])]
    # print(incidence_matrices[link].shape)

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
sclump = SClump(similarity_matrices, num_clusters=target_k)

# Run the algorithm!
print("running SClump...")
# labels, learned_similarity_matrix, metapath_weights = sclump.run(verbose=True)
# Run limited iterations:

similarity_matrix, metapath_weights = sclump.optimize(num_iterations=10, verbose=True)
labels = sclump.cluster(similarity_matrix)
metapath_weights_dict = {metapath: metapath_weights[index] for metapath, index in sclump.metapath_index.items()}

# print results
print("metapath weights:", metapath_weights)
if args.het_source == "degree":
    print("rand score:", rand_score(labels, target_clusters[type_lists[clustered_type],1]))
    print(labels)
elif args.het_source == "clustering": 
    print("rand scores of preclusterings:", 
          [rand_score(clustering_labels[i], target_clusters[:,1]) for i in range(args.num_clusterings)])
    print("rand score, sclump:", rand_score(labels, target_clusters[:,1]))
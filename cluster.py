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
import matplotlib.pyplot as plt


VARY_K = True
VARY_ALGORITHM = False

def save_graph(graph,file_name):
    #initialze Figure
    plt.figure(num=None, figsize=(20, 20), dpi=80)
    plt.axis('off')
    fig = plt.figure(1)
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph,pos)
    nx.draw_networkx_edges(graph,pos)
    # nx.draw_networkx_labels(graph,pos)

    cut = 1.00
    xmax = cut * max(xx for xx, yy in pos.values())
    ymax = cut * max(yy for xx, yy in pos.values())
    plt.xlim(-xmax, xmax)
    plt.ylim(-ymax, ymax)

    plt.savefig(file_name,bbox_inches="tight")
    del fig


def run_louvain(G, k):
    partition = community.best_partition(G)
    cluster_list = []
    i = 0
    while i in partition.keys():
        cluster_list.append(partition[i])
        i+=1
    print(len(cluster_list))
    return cluster_list

def run_metis(G, k):
    _, cluster_list = metis.part_graph(G, k)
    print(len(cluster_list))
    return cluster_list

def run_spectral_clustering(G, k):
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
a.add_argument("--sclump-iterations", help="number of iterations to run SClump for",
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

# naive clustering
print("performing naive clustering...")
clustering = SpectralClustering(n_clusters=target_k, affinity='precomputed', 
                                        random_state=random.randint(0,1e9))
clustering.fit(A)
naive_labels = clustering.labels_
print("rand score, naive clustering:", np.round(rand_score(naive_labels, target_clusters[:,1]), 3))

print(naive_labels)
naive_labels_attr = {}
for idx, naive_label in enumerate(naive_labels):
    naive_labels_attr[idx] = naive_label
nx.set_node_attributes(G, naive_labels_attr, "naive_spectral_clustering")


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
    ks = []
    for i in tqdm(range(args.num_clusterings)):
        if VARY_K:
            k = (target_k//args.num_clusterings)*(i+1)
            ks.append(k)
        else: 
            k = target_k
            ks.append(k)
        
        if not VARY_ALGORITHM:
            labels = run_spectral_clustering(G, k)
            labels_attr = {}
            for idx, label in enumerate(labels):
                labels_attr[idx] = label
            nx.set_node_attributes(G, labels_attr, "k_{k:n}_spectral_clustering".format(k=k))
        else:
            labels = clustering_methods[i](G, k)

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
    print("rand scores between preclusterings:\n", np.round(rand_scores, 3))

    # recompute adjacency matrix
    A = nx.adjacency_matrix(G_augmented).toarray()
    print("k's:", ks)

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
sclump = SClump(similarity_matrices, num_clusters=target_k)

# Run the algorithm!
print("running SClump...")
# labels, learned_similarity_matrix, metapath_weights = sclump.run(verbose=True)
# Run limited iterations:


similarity_matrix, metapath_weights = sclump.optimize(num_iterations=args.sclump_iterations, verbose=True)
metapath_weights = np.round(metapath_weights, 3)
labels = sclump.cluster(similarity_matrix)
sclump_labels_attr = {}
for idx, label in enumerate(labels):
    sclump_labels_attr[idx] = label
nx.set_node_attributes(G, sclump_labels_attr, "sclump_clustering")

metapath_weights_dict = {metapath: metapath_weights[index] for metapath, index in sclump.metapath_index.items()}

# print results
print("metapath weights:", metapath_weights_dict)
if args.het_source == "degree":
    print("rand score:", np.round(rand_score(labels, target_clusters[type_lists[clustered_type],1]), 3))
    print(labels)
elif args.het_source == "clustering": 
    print("rand scores of preclusterings:", 
          [np.round(rand_score(clustering_labels[i], target_clusters[:,1]), 3) for i in range(args.num_clusterings)])
    print("rand score, naive clustering:", np.round(rand_score(naive_labels, target_clusters[:,1]), 3))
    print("rand score, sclump:", np.round(rand_score(labels, target_clusters[:,1]), 3))

nx.write_graphml(G, "k_clusterings_sclump.graphml")
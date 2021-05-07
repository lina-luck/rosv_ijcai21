"""
Knowledge graph dataset for Relational-GCN
"""

from __future__ import print_function
from __future__ import absolute_import
import numpy as np
import scipy.sparse as sp
import os
import pandas as pd


np.random.seed(123)

class RGCNUnaryDataset(object):
    """RGCN Unary Classification dataset

    Each instance of unary template is regarded as a node
    its label is the template it satisfying

     An object of this class has 11 member attributes needed for unary
    classification:

    num_nodes: int
        number of entities of knowledge base
    num_rels: int
        number of relations (including reverse relation) of knowledge base
    num_classes: int
        number of classes/labels that of entities in knowledge base
    edge_src: numpy.array
        source node ids of all edges
    edge_dst: numpy.array
        destination node ids of all edges
    edge_type: numpy.array
        type of all edges
    edge_norm: numpy.array
        normalization factor of all edges
    labels: numpy.array
        labels of node entities
    train_idx: numpy.array
        ids of entities used for training
    valid_idx: numpy.array
        ids of entities used for validation
    test_idx: numpy.array
        ids of entities used for testing
    features: numpy.array
        input features of nodes

    When loading data, besides specifying dataset name, user can provide two
    optional arguments:

    Parameters
    ----------
    bfs_level: int
        prune out nodes that are more than ``bfs_level`` hops away from
        labeled nodes, i.e., nodes won't be touched during propagation. If set
        to a number less or equal to 0, all nodes will be retained.
    relabel: bool
        After pruning, whether or not to relabel all nodes with consecutive
        node ids
    """

    def __init__(self, name, type):
        self.name = name
        self.type = type
        self.dir = os.path.join(os.path.abspath('.'), 'dataset')
        self.dir = os.path.join(self.dir, self.type)
        self.dir = os.path.join(self.dir, self.name)

    def load(self, bfs_level=2, relabel=False):
        self.num_nodes, edges, self.num_rels, self.labels, labeled_nodes_idx, self.features = _load_data(self.name, self.dir)

        # bfs to reduce edges
        if bfs_level > 0:
            print("removing nodes that are more than {} hops away".format(bfs_level))
            row, col, edge_type = edges.transpose()
            A = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(self.num_nodes, self.num_nodes))
            bfs_generator = _bfs_relational(A, labeled_nodes_idx)
            lvls = list()
            lvls.append(set(labeled_nodes_idx))
            for _ in range(bfs_level):
                lvls.append(next(bfs_generator))
            to_delete = list(set(range(self.num_nodes)) - set.union(*lvls))
            eid_to_delete = np.isin(row, to_delete) + np.isin(col, to_delete)
            eid_to_keep = np.logical_not(eid_to_delete)
            self.edge_src = row[eid_to_keep]
            self.edge_dst = col[eid_to_keep]
            self.edge_type = edge_type[eid_to_keep]

            if relabel:  # False, so not run
                uniq_nodes, edges = np.unique((self.edge_src, self.edge_dst), return_inverse=True)
                self.edge_src, self.edge_dst = np.reshape(edges, (2, -1))
                node_map = np.zeros(self.num_nodes, dtype=int)
                self.num_nodes = len(uniq_nodes)
                node_map[uniq_nodes] = np.arange(self.num_nodes)
                self.labels = self.labels[uniq_nodes]
                self.train_idx = node_map[self.train_idx]
                self.test_idx = node_map[self.test_idx]
                print("{} nodes left".format(self.num_nodes))
        else:
            self.edge_src, self.edge_dst, self.edge_type = edges.transpose()

        # normalize by src degree, compute degrees according to edge_type
        _, inverse_index, count = np.unique((self.edge_dst, self.edge_type), axis=1, return_inverse=True, return_counts=True)
        degrees = count[inverse_index]  #c_{i,r} for each relation type
        self.edge_norm = np.ones(len(self.edge_dst), dtype=np.float32) / degrees.astype(np.float32)

        # convert to pytorch label format
        self.num_classes = self.labels.shape[1]
        #self.labels = np.argmax(self.labels, axis=1)


class RGCNBinaryDataset(object):
    """RGCN link prediction dataset

    Each binary template is regard as a binary relation between two entities (nodes)

    An object of this class has 5 member attributes needed for link
    prediction:

    num_nodes: int
        number of entities of rule base
    num_rels: int
        number of relations (including reverse relation) of rule base
    data: numpy.array
        all relation triplets (src, rel, dst)
    relation_features: dateframe

    Usually, user don't need to directly use this class. Instead, DGL provides
    wrapper function to load data (see example below).

    """
    def __init__(self, name):
        self.name = name
        self.dir = os.path.join(os.path.abspath('.'), 'binary_data')
        self.dir = os.path.join(self.dir, self.name)

    def load(self):
        entity_path = os.path.join(self.dir, 'nodes.dict')
        relation_path = os.path.join(self.dir, 'edges.dict')
        data_path = os.path.join(self.dir, 'binary_template_triples.txt')  # all triples
        node_feature_path = os.path.join(self.dir, 'node_features.csv')
        relation_feature_path = os.path.join(self.dir, 'relation_features.csv')
        self.node_features = pd.read_csv(node_feature_path, sep=',', encoding='utf-8')
        self.relation_features = pd.read_csv(relation_feature_path, sep=',', encoding='utf-8')
        entity_dict = _read_dictionary(entity_path)
        relation_dict = _read_dictionary(relation_path)
        self.complete_data = np.array(_read_triplets_as_list(data_path, entity_dict, relation_dict))  # (n_id, r_id, n_id)
        self.num_nodes = len(entity_dict)
        print("# entities: {}".format(self.num_nodes))
        self.num_rels = len(relation_dict)
        print("# relations: {}".format(self.num_rels))
        print("# edges: {}".format(len(self.complete_data)))   # total number of edges


def load_unary(dataset, feature_type, bfs_level, relabel):
    data = RGCNUnaryDataset(dataset, feature_type)
    data.load(bfs_level, relabel)
    return data


def load_binary(dataset):
    data = RGCNBinaryDataset(dataset)
    data.load()
    return data


def _sp_row_vec_from_idx_list(idx_list, dim):
    """Create sparse vector of dimensionality dim from a list of indices."""
    shape = (1, dim)
    data = np.ones(len(idx_list))
    row_ind = np.zeros(len(idx_list))
    col_ind = list(idx_list)
    return sp.csr_matrix((data, (row_ind, col_ind)), shape=shape)

def _get_neighbors(adj, nodes):
    """Takes a set of nodes and a graph adjacency matrix and returns a set of neighbors."""
    sp_nodes = _sp_row_vec_from_idx_list(list(nodes), adj.shape[1])
    sp_neighbors = sp_nodes.dot(adj)
    neighbors = set(sp.find(sp_neighbors)[1])  # convert to set of indices
    return neighbors

def _bfs_relational(adj, roots):
    """
    BFS for graphs with multiple edge types. Returns list of level sets.
    Each entry in list corresponds to relation specified by adj_list.
    """
    visited = set()
    current_lvl = set(roots)

    next_lvl = set()

    while current_lvl:

        for v in current_lvl:
            visited.add(v)

        next_lvl = _get_neighbors(adj, current_lvl)
        next_lvl -= visited  # set difference

        yield next_lvl

        current_lvl = set.union(next_lvl)


def _load_sparse_csr(filename):
    loader = np.load(filename)
    return sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape=loader['shape'], dtype=np.float32)


def _save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def _load_data(dataset_str='wine', dataset_path=None):
    """

    :param dataset_str:
    :param rel_layers:
    :param limit: If > 0, will only load this many adj. matrices
        All adjacencies are preloaded and saved to disk,
        but only a limited a then restored to memory.
    :return:
    """

    print('Loading dataset', dataset_str)
    fea_file = 'node_features.csv'
    feature_file = os.path.join(dataset_path, fea_file)  # node_features: word embedding + analogy space
    edge_file = os.path.join(dataset_path, 'edges_subset.npz')    # {edges: (s, t, r) / n: num_nodes / nrel : num_rels}
    labels_file = os.path.join(dataset_path, 'labels_subset.npz')  # n * l sparse matrix

    # load node features
    node_feature = pd.read_csv(feature_file, sep=',', encoding='utf-8')

    # load precomputed adjacency matrix and labels
    all_edges = np.load(edge_file)
    num_node = all_edges['n'].item()
    edge_list = all_edges['edges']
    num_rel = all_edges['nrel'].item()

    print('Number of nodes: ', num_node)
    print('Number of edges: ', len(edge_list))
    print('Number of relations: ', num_rel)

    labels = _load_sparse_csr(labels_file)
    labeled_nodes_idx = list(labels.nonzero()[0])

    print('Number of classes: ', labels.shape[1])

    return num_node, edge_list, num_rel, labels, labeled_nodes_idx, node_feature


# key = template name, value = id
def _read_dictionary(filename):
    d = {}
    with open(filename, 'r+') as f:
        for line in f:
            line = line.strip().split('\t')
            d[line[1]] = int(line[0])
    return d

# key = id, value = template name
def read_dictionary(filename):
    d = {}
    with open(filename, 'r+') as f:
        for line in f:
            line = line.strip().split('\t')
            d[int(line[0])] = line[1]
    return d

def _read_triplets(filename):
    with open(filename, 'r+') as f:
        for line in f:
            processed_line = line.strip().split('\t')
            yield processed_line

def _read_triplets_as_list(filename, entity_dict, relation_dict):
    l = []
    for triplet in _read_triplets(filename):
        s = entity_dict[triplet[0]]
        r = relation_dict[triplet[1]]
        o = entity_dict[triplet[2]]
        l.append([s, r, o])
    return l



import  numpy as np
import  pickle as pkl
import  networkx as nx
import  scipy.sparse as sp
# from    scipy.sparse.linalg.eigen.arpack import eigsh
from scipy.sparse.linalg import eigsh
import  sys
import torch
from collections import Counter
from npz_utils import *
import torch_geometric.transforms as T
from torch_geometric.data import Data
import scipy
import torch_geometric


def parse_index_file(filename):
    """
    Parse index file.
    """
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """
    Create mask.
    """
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool_)




def load_data_ood(dataset_str):
    if dataset_str in  ["cora", "citeseer", "pubmed"]:
        return load_citation_ood(dataset_str)
    elif dataset_str in ["amazon_electronics_computers", "amazon_electronics_photo", "ms_academic_cs", "ms_academic_phy"]:
        return load_amazon_ood(dataset_str)
    else:
        raise ValueError('unknown dataset')


def load_citation_ood(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) 

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    labels = np.argmax(labels, axis=1)
    labels = torch.LongTensor(labels)

    if dataset_str == 'cora':
        index = labels > 3
        labels[index] = -1


    if dataset_str == 'citeseer':
        index = labels > 2
        labels[index] = -1

    if dataset_str == 'pubmed':
        index = labels > 1
        labels[index] = -1


    features = preprocess_features(features) 
    # adj = preprocess_adj(adj)


    with np.load(f'adj/{dataset_str}_adj.npz') as data:
        value = data['value']
        value = np.clip(value,0,1.0)
        adj = sp.coo_matrix((value,(data['row'],data['col'])),shape=data['shape'])
        adj = preprocess_adj(adj)

    features = torch.FloatTensor(np.array(features.todense()))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor([i for i in list(idx_train) if labels[i] >= 0])
    idx_val = torch.LongTensor([i for i in list(idx_val) if labels[i] >= 0])
    idx_test_id = torch.LongTensor([i for i in list(idx_test) if labels[i] >= 0])

    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val,idx_test_id, idx_test



def load_amazon_ood(dataset_str):  
    adj, features, labels = get_dataset("data/{}.npz".format(dataset_str), True)
    features = features.toarray() 
    adj = adj.tocoo()
    
    random_state = np.random.RandomState(42)
    y_train, y_val, y_test, train_mask, val_mask, test_mask, _ = get_train_val_test_split_ood(random_state, labels, train_size=20*labels.shape[1], val_size=30*labels.shape[1])

    test_mask = np.array(1 - train_mask - val_mask, dtype=bool)

    y_train = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]

    labels = np.argmax(labels, axis=1)
    labels = torch.LongTensor(labels)

    if dataset_str == 'amazon_electronics_photo':
        index = labels > 3
        labels[index] = -1

    if dataset_str == 'amazon_electronics_computers':
        index = labels > 4
        labels[index] = -1

    if dataset_str == 'ms_academic_phy':
        index = labels > 2
        labels[index] = -1

    if dataset_str == 'ms_academic_cs':
        index = labels > 7
        labels[index] = -1


    # adj = preprocess_adj(adj)

    with np.load(f'adj/{dataset_str}_adj.npz') as data:
        value = data['value']
        value = np.clip(value,0,1.0)
        adj = sp.coo_matrix((value,(data['row'],data['col'])),shape=data['shape'])
        adj = preprocess_adj(adj)


    features = torch.FloatTensor(features)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = np.array([i for i, ele in enumerate(train_mask) if ele == True and labels[i] >= 0])
    idx_val = np.array([i for i, ele in enumerate(val_mask) if ele == True and labels[i] >= 0])
    idx_test_id = np.array([i for i, ele in enumerate(test_mask) if ele == True and labels[i] >= 0])
    idx_test = np.array([i for i, ele in enumerate(test_mask) if ele == True])


    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test_id = torch.LongTensor(idx_test_id)
    idx_test = torch.LongTensor(idx_test)


    return adj, features, labels, idx_train, idx_val,idx_test_id, idx_test



def filter_idx(idx, labels,number, random_state):
    unique_label = set(labels.tolist())
    new_idx = []

    for label in unique_label:
        if label >= 0 :
            tmp_idx = idx[labels[idx]==label]
            tmp_idx = random_state.permutation(tmp_idx)[:number]
            new_idx += tmp_idx.tolist()

    new_idx = torch.LongTensor(new_idx)

    return new_idx



def is_binary_bag_of_words(features):
    features_coo = features.tocoo()
    return all(single_entry == 1.0 for _, _, single_entry in zip(features_coo.row, features_coo.col, features_coo.data))


def get_dataset(data_path, standardize):
    dataset_graph = load_npz_to_sparse_graph(data_path)

    if standardize:
        dataset_graph = dataset_graph.standardize()
    else:
        dataset_graph = dataset_graph.to_undirected()
        dataset_graph = eliminate_self_loops(dataset_graph)

    adj_matrix, attr_matrix, labels = dataset_graph.unpack()

    labels = binarize_labels(labels)
    # convert to binary bag-of-words feature representation if necessary
    if not is_binary_bag_of_words(attr_matrix):
        attr_matrix = to_binary_bag_of_words(attr_matrix)

    return adj_matrix, attr_matrix, labels



def get_train_val_test_split_ood(random_state,
                             labels,
                             train_examples_per_class=None, val_examples_per_class=None,
                             test_examples_per_class=None,
                             train_size=None, val_size=None, test_size=None):
    num_samples, num_classes = labels.shape
    remaining_indices = list(range(num_samples))

    if train_examples_per_class is not None:
        train_indices = sample_per_class(random_state, labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = random_state.choice(remaining_indices, train_size, replace=False)
        train_mask = sample_mask(train_indices, labels.shape[0])

    if val_examples_per_class is not None:
        val_indices = sample_per_class(random_state, labels, val_examples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(remaining_indices, val_size, replace=False)
        val_mask = sample_mask(val_indices, labels.shape[0])

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(random_state, labels, test_examples_per_class,
                                        forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(remaining_indices, test_size, replace=False)
        test_mask = sample_mask(test_indices, labels.shape[0])
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_mask = sample_mask(test_indices, labels.shape[0])

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert len(np.concatenate((train_indices, val_indices, test_indices))) == num_samples

    if train_examples_per_class is not None:
        train_labels = labels[train_indices, :]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices, :]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices, :]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return y_train, y_val, y_test, train_mask, val_mask, test_mask, train_indices


def sample_per_class(random_state, labels, num_examples_per_class, forbidden_indices=None):
    num_samples, num_classes = labels.shape
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index, class_index] > 0.0:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])


def weighted_adj(adj,adj_weight,labels):
    dense_adj = adj.todense().astype('float32')
    row,col = dense_adj.nonzero()

    for i,j in zip(row,col):
        if labels[i] < 0 or labels[j] < 0:
            dense_adj[i][j] = adj_weight


    weighted_adj = np.asmatrix(dense_adj)
    return weighted_adj


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def sparse_to_tuple(sparse_mx):
    """
    Convert sparse matrix to tuple representation.
    """
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """
    Row-normalize feature matrix and convert to tuple representation
    """
    rowsum = np.array(features.sum(1)) # get sum of each row, [2708, 1]
    r_inv = np.power(rowsum, -1).flatten() # 1/rowsum, [2708]
    r_inv[np.isinf(r_inv)] = 0. # zero inf data
    r_mat_inv = sp.diags(r_inv) # sparse diagonal matrix, [2708, 2708]
    features = r_mat_inv.dot(features) # D^-1:[2708, 2708]@X:[2708, 2708]
    return features # [coordinates, data, shape], []


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return (d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)).tocoo()




def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized

def preprocess_refined_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj)
    return adj_normalized


def chebyshev_polynomials(adj, k):
    """
    Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation).
    """
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)



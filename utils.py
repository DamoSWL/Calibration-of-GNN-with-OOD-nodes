import numpy as np
import scipy.sparse as sp
import torch
import random
import matplotlib.pyplot as plt

def manual_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



def plot_acc_calibration(idx_test, output, labels,seed,dataset, n_bins=10, title=None):
        output = torch.exp(output)
        pred_label = torch.max(output[idx_test], 1)[1]
        p_value = torch.max(output[idx_test], 1)[0]
        ground_truth = labels[idx_test]
        confidence_all, confidence_acc = np.zeros(n_bins), np.zeros(n_bins)
        for index, value in enumerate(p_value):
            #value -= suboptimal_prob[index]
            interval = int(value / (1 / n_bins) -0.0001)
            confidence_all[interval] += 1
            if pred_label[index] == ground_truth[index]:
                confidence_acc[interval] += 1
        for index, value in enumerate(confidence_acc):
            if confidence_all[index] == 0:
                confidence_acc[index] = 0
            else:
                confidence_acc[index] /= confidence_all[index]

        start = np.around(1/n_bins/2, 3)
        step = np.around(1/n_bins, 3)
        plt.figure(figsize=(5, 4))
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams["font.weight"] = "bold"

        plt.bar(np.around(np.arange(start, 1.0, step), 3),
                np.around(np.arange(start, 1.0, step), 3), alpha=0.6, width=0.09, color='lightcoral', label='Expected')
        plt.bar(np.around(np.arange(start, 1.0, step), 3), confidence_acc,
                alpha=0.6, width=0.09, color='dodgerblue', label='Outputs')       
        plt.plot([0,1], [0,1], ls='--',c='k')
        plt.xlabel('Confidence', fontsize=16)
        plt.ylabel('Accuracy', fontsize=16)
        plt.tick_params(labelsize=13)
        plt.xlim(0, 1.0)
        plt.ylim(0, 1.0)
        #title = 'Uncal. - Cora - 20 - GCN'
        plt.title(title, fontsize=16, fontweight="bold")
        plt.legend(fontsize=16)
        plt.savefig('images/' + str(seed)+ '_' + 'gcn' + '_' + dataset + '.png' , format='png', dpi=300,
                    pad_inches=0, bbox_inches = 'tight')
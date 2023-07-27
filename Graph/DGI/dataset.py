import numpy as np
import scipy.sparse as sp
import torch

def sparse_mx_to_torch(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_adj_feature(adj, feature):
    
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    adj = adj + sp.eye(adj.shape[0])
    
    adj = sparse_mx_to_torch(adj)
    
    rowsum = np.sum(feature, axis=1)
    rowsum_diag = np.diag(rowsum)
    rowsum_inv = np.power(rowsum_diag, -1)
    rowsum_inv[np.isinf(rowsum_inv)] = 0.0
    feature = np.dot(rowsum_inv, feature)
    
    feature = torch.FloatTensor(feature[np.newaxis])
    
    return adj, feature
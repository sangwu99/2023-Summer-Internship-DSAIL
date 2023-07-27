from torch_geometric.datasets import Planetoid

import numpy as np 
import scipy.sparse as sp

from scipy.sparse import coo_matrix, csr_matrix


def load_dataset(args):
    
    if args.dataset_name == 'Cora':
        dataset = Planetoid(root='/tmp/Cora', name='Cora')
    elif args.dataset_name == 'Pubmed':
        dataset = Planetoid(root='/tmp/Pubmed', name='Pubmed')
    elif args.dataset_name == 'Citeseer':
        dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')
    else:
        raise ValueError('Dataset name must be one of Cora, Pubmed, Citeseer')
    
    graph = dataset[0]
    adj = csr_matrix((np.ones(graph.edge_index.shape[1]), (graph.edge_index[0].numpy(), graph.edge_index[1].numpy())))
    feature = graph.x.numpy()
    
    return adj, feature, graph
import torch
from torch.utils.data import Dataset, DataLoader

class FB15k(Dataset):
    def __init__(self, graph, idx):
        self.head, self.tail = graph.edges._graph.find_edges(idx)
        self.label = graph.edata['etype'][idx]
        self.num_entities = graph.num_nodes()
        
    def __len__(self):
        return len(self.head)
    
    def __getitem__(self, idx):
        return self.head[idx], self.label[idx], self.tail[idx] 


def split_dataset(graph, batch_size=128):
    train_mask, val_mask, test_mask = graph.edata['train_mask'], graph.edata['val_mask'], graph.edata['test_mask']
    
    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
    val_idx = torch.nonzero(val_mask, as_tuple=False).squeeze()
    test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze()
    
    train_dataset = FB15k(graph, train_idx)
    val_dataset = FB15k(graph, val_idx)
    test_dataset = FB15k(graph, test_idx)
    
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)
    
    
    return train_dataloader, val_dataloader, test_dataloader
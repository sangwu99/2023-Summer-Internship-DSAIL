from dgl.data import FB15kDataset

def load_graph():
    dataset = FB15kDataset()
    return dataset[0]
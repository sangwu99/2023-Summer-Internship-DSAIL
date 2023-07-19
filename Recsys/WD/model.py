import torch
import torch.nn as nn

class Wide(nn.Module):
    def __init__(self, wide_dim, output_dim):
        super(Wide, self).__init__()
        self.linear = nn.Linear(wide_dim, output_dim)
        
    def forward(self, x):
        output = self.linear(x)
        return output
    
class Deep(nn.Module):
    def __init__(self, embedding_input, factor_dim, layer_num, hidden_dim, output_dim):
        super(Deep, self).__init__()
        self.embedding_input = embedding_input
        self.factor_dim = factor_dim
        self.layer_num = layer_num
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        for idx, val in enumerate(self.embedding_input):
            setattr(self, 'embedding_{}'.format(idx), nn.Embedding(val, self.factor_dim))
        
        self.dense_layers = self.dense()
        
    def dense(self):
        dense = []
        self.factor_dim *= len(self.embedding_input)
        dense.append(nn.Linear(self.factor_dim, self.hidden_dim[0], bias= True))
        dense.append(nn.ReLU())
        for idx in range(self.layer_num-1):
            dense.append(nn.Linear(self.hidden_dim[idx], self.hidden_dim[idx+1], bias= True))
            dense.append(nn.ReLU())
        dense.append(nn.Linear(self.hidden_dim[-1], self.output_dim))
        
        return nn.Sequential(*dense)
    
    def forward(self, x):
        output = [getattr(self, 'embedding_{}'.format(idx))(x[:,idx]) for idx, val in enumerate(self.embedding_input)]
        output = torch.cat(output, 1)
        
        output = self.dense_layers(output)
        
        return output
    
class WideAndDeep(nn.Module):
    def __init__(self, wide_dim, embedding_input, factor_dim, layer_num, hidden_dim, output_dim):
        super(WideAndDeep, self).__init__()
        
        self.wide = Wide(wide_dim, output_dim)
        self.deep = Deep(embedding_input, factor_dim, layer_num, hidden_dim, output_dim)
        
    def forward(self, wide, deep):
        wide_component = self.wide(wide)
        deep_component = self.deep(deep)
        return torch.sigmoid(torch.add(wide_component, deep_component))
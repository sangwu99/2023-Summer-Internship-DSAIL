import torch 
import torch.nn as nn 
import torch.nn.functional as F

class AutoRec(nn.Module):
    def __init__(self, num_users, num_items, hidden_dim, is_item = True):
        super(AutoRec, self).__init__()
        if is_item == True:
            self.hidden_dim = [num_users] + hidden_dim
        else:
            self.hidden_dim = [num_items] + hidden_dim
        self.encoder = nn.ModuleList([nn.Linear(self.hidden_dim[idx], self.hidden_dim[idx+1])
                                        for idx in range(len(self.hidden_dim)-1)])
        self.decoder = nn.ModuleList([nn.Linear(self.hidden_dim[idx], self.hidden_dim[idx-1])
                                        for idx in range(len(self.hidden_dim)-1, 0, -1)])
        self.init_weights()
        
    def init_weights(self):
        for layer in self.encoder:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        for layer in self.decoder:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            
    def forward(self, x):
        for layer in self.encoder:
            x = F.relu(layer(x))
        for idx in range(len(self.decoder)):
            if idx == len(self.decoder) -1:
                x = torch.sigmoid(self.decoder[idx](x))
            else:
                x = F.relu(self.decoder[idx](x))
        return x
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        
        self.hidden_dim = [input_dim] + hidden_dim
        self.encoder = nn.ModuleList([nn.Linear(self.hidden_dim[idx], self.hidden_dim[idx+1]) 
                                      for idx in range(len(self.hidden_dim)-1)])
        self.mu = nn.Linear(self.hidden_dim[-1], latent_dim)
        self.logvar = nn.Linear(self.hidden_dim[-1], latent_dim)
        self.decoder = nn.ModuleList([nn.Linear(latent_dim, self.hidden_dim[-1])] + [nn.Linear(self.hidden_dim[idx], self.hidden_dim[idx-1]) 
                                                                                for idx in range(len(self.hidden_dim)-1, 0, -1)])
        
        self.init_weights()
        
    def init_weights(self):
        for layer in self.encoder:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        for layer in self.decoder:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)
        
    def reparameterization(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        
        return mu + eps * std
    
    def forward(self, x):
        for layer in self.encoder:
            x = F.relu(layer(x))
            
        mu = F.relu(self.mu(x))
        logvar = F.relu(self.logvar(x))
        z = self.reparameterization(mu, logvar)
        
        for idx in range(len(self.decoder)):
            if idx == len(self.decoder) -1: 
                z = F.sigmoid(self.decoder[idx](z))
            else:
                z = F.relu(self.decoder[idx](z))
        
        return z, mu, logvar
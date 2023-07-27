import torch 
import torch.nn as nn 
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCN, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.PRelu = nn.PReLU() 
        
        nn.init.xavier_uniform_(self.linear.weight.data)
        
    def forward(self, x, a):
        xtheta = self.linear(x)
        output = torch.unsqueeze(torch.spmm(a, torch.squeeze(xtheta, 0)), 0)
        output = self.PRelu(output)
        
        return output 
    
class Discriminator(nn.Module):
    def __init__(self, in1_features, in2_features, out_features):
        super(Discriminator, self).__init__()
        self.linear = nn.Bilinear(in1_features, in2_features, out_features)
        # self.logsigmoid = nn.LogSigmoid()
        
        nn.init.xavier_uniform_(self.linear.weight.data)
        
    def forward(self, hi, s):
        
        
        hiWs = self.linear(hi, s)
        # output = self.logsigmoid(hiWs)
        
        return hiWs
    
class Readout(nn.Module):
    def __init__(self):
        super(Readout, self).__init__()
        # self.logsigmoid = nn.LogSigmoid()
        
    def forward(self, H):
        output = torch.mean(H, dim=1)
        # output = self.logsigmoid(output)
        
        return output
    
class DGI(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super(DGI, self).__init__()
        self.GCN = GCN(in_features, hidden_dim)
        self.Discriminator = Discriminator(hidden_dim, hidden_dim, 1)
        self.Readout = Readout()
        self.Sigmoid = nn.Sigmoid()
        
    def forward(self, pos, neg, a):
        pos_H = self.GCN(pos, a)
        neg_H = self.GCN(neg, a)
        
        s = self.Readout(pos_H)
        s = self.Sigmoid(s) 
        s = torch.unsqueeze(s, 1).expand_as(pos_H)
        
        pos_score = self.Discriminator(pos_H, s).squeeze(2)
        neg_score = self.Discriminator(neg_H, s).squeeze(2) 
        logits = torch.cat((pos_score, neg_score), dim=1)
        
        return logits
    
    def get_embedding(self, x, a):
        return self.GCN(x, a)
import os 
import numpy as np 

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.optim import SGD

from sklearn.metrics.pairwise import cosine_similarity


class BaselineEstimates(nn.Module):
    def __init__(self, num_users, num_items, mu):
        super(BaselineEstimates, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.mu = mu
        
        self.user_biases = nn.Embedding(num_users, 1)
        self.item_biases = nn.Embedding(num_items, 1)
        
        self.user_biases.weight.data.normal_(0,1)
        self.item_biases.weight.data.normal_(0,1)
    
    def forward(self, user, item):
        bu = self.user_biases(user)
        bi = self.item_biases(item)
        
        rui = self.mu + torch.squeeze(bu) + torch.squeeze(bi)
        
        return rui
    
class NeighborhoodModel(nn.Module):
    def __init__(self, R, mu, k, device):
        super(NeighborhoodModel, self).__init__()
        self.R = R 
        self.k = k
        self.num_users, self.num_items = R.shape
        self.Base = BaselineEstimates(self.num_users, self.num_items, mu)
        self.item_weights = nn.Parameter(torch.normal(0,1,size=(self.num_items,self.num_items)))
        self.implicit_offset = nn.Parameter(torch.normal(0,1,size=(self.num_items,self.num_items)))
        self.mu = mu
        self.S = cosine_similarity(R.T)
        self.device = device
        
        self.get_top_k()
        self.get_implicit()
        
    def get_top_n_indices(self, list, n):
        sorted_indices = sorted(range(len(list)), key=lambda i: list[i], reverse=True)
        top_n_indices = sorted_indices[:n]
        
        return top_n_indices

    def get_top_k(self):
        self.similar_k = {}
        for item in range(self.num_items):
            self.similar_k[item] = self.get_top_n_indices(self.S[item], self.k)
            
    def get_implicit(self):
        self.implicit_data = {} 
        users, items = self.R.toarray().nonzero()
        for user, item in zip(users, items):
            if user not in self.implicit_data:
                self.implicit_data[user] = []
            self.implicit_data[user].append(item)
  
    def forward(self, user, item):
        bui = self.Base(user, item)
        user_idx = int(user)
        item_idx = int(item)
        
        sum_of_item_weights = 0
        sum_of_implicit_offset = 0
        num_k = 0
        
        self.used_items = self.implicit_data[user_idx]
        
        for implicit in self.implicit_data[user_idx]:
            if implicit in self.similar_k[item_idx]:
                implicit_tensor = torch.LongTensor([implicit]).to(self.device)
                num_k += 1
                
                with torch.no_grad():
                    buj = self.Base(user, implicit_tensor)
                    
                sum_of_item_weights += (int(self.R[user,implicit].data)-buj) * self.item_weights[item][0][implicit]
                sum_of_implicit_offset += self.implicit_offset[item][0][implicit]        
            
        norm = num_k ** -0.5

        rui = bui + norm * sum_of_item_weights + norm * sum_of_implicit_offset
        
        return rui
    
class AsymmetricSVD(nn.Module):
    def __init__(self, R, mu, F, device):
        super(AsymmetricSVD, self).__init__()
        self.num_users, self.num_items = R.shape
        self.Base = BaselineEstimates(self.num_users, self.num_items, mu)
        self.R = R 
        self.Q = nn.Embedding(self.num_items, F)
        self.X = nn.Embedding(self.num_items, F)
        self.Y = nn.Embedding(self.num_items, F)
        
        self.Q.weight.data.normal_(0, 1/F)
        self.X.weight.data.normal_(0, 1/F)
        self.Y.weight.data.normal_(0, 1/F)
        
        self.device = device
        self.get_implicit()
        
    def get_implicit(self):
        self.implicit_data = {} 
        users, items = self.R.toarray().nonzero()
        for user, item in zip(users, items):
            if user not in self.implicit_data:
                self.implicit_data[user] = []
            self.implicit_data[user].append(item)
        
    def forward(self, user, item):
        user_idx = int(user)
        
        bui = self.Base(user, item)
        Q_i = self.Q(item)
        
        sum_of_item_weights = 0
        sum_of_implicit_offset = 0
        
        for implicit in self.implicit_data[user_idx]:
            implicit_tensor = torch.LongTensor([implicit]).to(self.device)
            with torch.no_grad():
                buj = self.Base(user, implicit_tensor)
                
            sum_of_item_weights += (int(self.R[user,implicit].data) - buj) * self.X(implicit_tensor)
            sum_of_implicit_offset += self.Y(implicit_tensor)
            
        norm = len(self.implicit_data[user_idx]) ** -0.5        
        
        rui = bui + torch.sum(Q_i * (norm * (sum_of_item_weights + sum_of_implicit_offset)), dim = 1)
        
        return rui
    
class SVDPlusPlus(nn.Module):
    def __init__(self, R, mu, F, device, is_layer=False):
        super(SVDPlusPlus, self).__init__()
        self.is_layer = is_layer
        self.R = R 
        self.num_users, self.num_items = R.shape
        self.Base = BaselineEstimates(self.num_users, self.num_items, mu)
        
        self.user_embedding = nn.Embedding(self.num_users, F)
        self.item_embedding = nn.Embedding(self.num_items, F)
        
        self.Y = nn.Embedding(self.num_items, F)
        self.device = device
        
        self.user_embedding.weight.data.normal_(0,1/F)
        self.item_embedding.weight.data.normal_(0,1/F)
        self.Y.weight.data.normal_(0,1/F)
        self.get_implicit()
        
    def get_implicit(self):
        self.implicit_data = {} 
        users, items = R.toarray().nonzero()
        for user, item in zip(users, items):
            if user not in self.implicit_data:
                self.implicit_data[user] = []
            self.implicit_data[user].append(item)
        
    def forward(self, user, item):
        user_idx = int(user)
        
        bui = self.Base(user, item)
        
        P_u = self.user_embedding(user)
        Q_i = self.item_embedding(item)
        
        sum_of_implicit_offset = 0
        for implicit in self.implicit_data[user_idx]:
            implicit_tensor = torch.LongTensor([implicit]).to(self.device)
            sum_of_implicit_offset += self.Y(implicit_tensor)
        
        norm = len(self.implicit_data[user_idx]) ** -0.5
        
        if self.is_layer:
            rui = torch.sum(P_u * (Q_i + norm * sum_of_implicit_offset), dim = 1)
        else:
            rui = bui + torch.sum(P_u * (Q_i + norm * sum_of_implicit_offset), dim = 1)
        
        return rui
    
class IntergratedModel(nn.Module):
    def __init__(self, R, mu, F, k, device):
        super(IntergratedModel, self).__init__()
        self.neighbor = NeighborhoodModel(R, mu, k, device)
        self.SVD = SVDPlusPlus(R, mu, F, device, is_layer=True)
        
        self.neighbor.get_implicit()
        self.neighbor.get_top_k()
        self.SVD.get_implicit()
        
    def forward(self, user, item):
        rui = self.neighbor(user, item) + self.SVD(user, item)
        
        return rui
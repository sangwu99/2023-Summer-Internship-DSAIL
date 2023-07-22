import torch 
import torch.nn as nn

class SoRec(nn.Module):
    def __init__(self, num_users, num_items, num_factors):
        super(SoRec, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_factors = num_factors
        
        self.user_embeddings = nn.Embedding(num_users, num_factors)
        self.item_embeddings = nn.Embedding(num_items, num_factors)
        self.social_embeddings = nn.Embedding(num_users, num_factors)
        
        self.user_embeddings.weight.data.normal_(0,1)
        self.item_embeddings.weight.data.normal_(0,1)
        
    def rating(self, user, item):
        user_embedding = self.user_embeddings(user)
        item_embedding = self.item_embeddings(item)
        
        return torch.sigmoid(torch.sum(user_embedding * item_embedding, dim=1))
    
    def social(self, user, target_user):
        user_embedding = self.user_embeddings(user)
        target_user_embedding = self.social_embeddings(target_user)
        
        return torch.sigmoid(torch.sum(user_embedding * target_user_embedding, dim=1))
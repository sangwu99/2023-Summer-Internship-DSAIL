import torch
import numpy as np
from torch.utils.data import Dataset

class Rating(Dataset):
    def __init__(self, df):
        self.df = df
        self.users = self.df['user'].to_numpy(dtype=np.int64)
        self.items = self.df['item'].to_numpy(dtype=np.int64)
        self.ratings = self.df['rating'].to_numpy(dtype=np.int64)
        
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        user = self.users[index]
        item = self.items[index]
        rating = self.ratings[index]
        
        return user, item, rating
    
class Social(Dataset):
    def __init__(self, df):
        self.df = df
        self.users = self.df['user'].to_numpy(dtype=np.int64)
        self.target_users = self.df['target_user'].to_numpy(dtype=np.int64)
        self.trust_values = self.df['trust_value'].to_numpy(dtype=np.int64)
        
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        user = self.users[index]
        target_user = self.target_users[index]
        trust_value = self.trust_values[index]
        
        return user, target_user, trust_value
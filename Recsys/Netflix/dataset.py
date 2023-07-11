from torch.utils.data import Dataset
import pandas as pd 

class Netflix(Dataset):
    def __init__(self, df):
        self.df = df
        self.users = self.df['Cust_ID'].values
        self.items = self.df['Movie_Id'].values
        self.ratings = self.df['Rating'].values
        
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        user = self.users[index]
        item = self.items[index]
        rating = self.ratings[index]
        
        return user, item, rating
    
class Temporal_Netflix(Dataset):
    def __init__(self, df):
        self.df = df
        self.users = self.df['Cust_ID'].values
        self.items = self.df['Movie_Id'].values
        self.ratings = self.df['Rating'].values
        self.time = self.df['bins'].values
        
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        user = self.users[index]
        item = self.items[index]
        time = self.time[index]

        rating = self.ratings[index]
        
        return user, item,time, rating
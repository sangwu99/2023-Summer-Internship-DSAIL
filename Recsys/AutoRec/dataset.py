import torch 
from torch.utils.data import Dataset

class AutoRecDataset(Dataset):
    def __init__(self, user_list, item_list, rating_list, num_user, num_item, is_item=True):
        super(AutoRecDataset, self).__init__()
        self.is_item = is_item
        self.user_list = user_list
        self.item_list = item_list
        self.rating_list = rating_list
        self.num_user = num_user
        self.num_item = num_item
        
        self.make_mat()
        
    def make_mat(self):
        if self.is_item==True:
            self.matrix = torch.zeros(self.num_item, self.num_user)
            for user, item, rating in zip(self.user_list, self.item_list, self.rating_list):
                self.matrix[item, user] = rating
        else:
            self.matrix = torch.zeros(self.num_user, self.num_item)
            for user, item, rating in zip(self.user_list, self.item_list, self.rating_list):
                self.matrix[user, item] = rating
    
    def __len__(self):
        if self.is_item==True:
            return self.num_item
        else:
            return self.num_user
    
    def __getitem__(self, idx):
        return self.matrix[idx]
        
        
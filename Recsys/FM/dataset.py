import numpy as np 
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset , DataLoader

def load_data_split(args, df):
    train_X, test_X, train_y, test_y = train_test_split(
        df.loc[:, df.columns != 'rating'], df['rating'], test_size=args.test_size, random_state=args.seed)
    
    train_dataset_fm = TensorDataset(torch.Tensor(np.array(train_X)), torch.Tensor(np.array(train_y)))
    test_dataset_fm = TensorDataset(torch.Tensor(np.array(test_X)), torch.Tensor(np.array(test_y)))
    
    train_dataloader_fm = DataLoader(train_dataset_fm, batch_size=args.batch_size, shuffle=True)
    test_dataloader_fm = DataLoader(test_dataset_fm, batch_size=args.batch_size, shuffle=True)
    
    return train_dataloader_fm, test_dataloader_fm
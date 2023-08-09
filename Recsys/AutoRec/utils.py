import os 
import pandas as pd 
import numpy as np 
import torch

from sklearn.model_selection import train_test_split

def preprocess_rating(x):
    if x == 0:
        return -1 
    else:
        return (x-1) / 4

def load_data(args):
    df = pd.read_csv(os.path.join(args.dpath,'u.data'), sep='\t', header=None)
    df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    
    user2idx = {j:i for i,j in enumerate(df.user_id.unique())}
    item2idx = {j:i for i,j in enumerate(df.item_id.unique())}

    df['user_id'] = df['user_id'].map(user2idx)
    df['item_id'] = df['item_id'].map(item2idx)
    df['rating'] = df['rating'].apply(preprocess_rating)
    
    num_items = df.item_id.nunique()
    num_users = df.user_id.nunique()

    train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=42)
    train_user, train_item, train_rating = train_df.user_id.values, train_df.item_id.values, train_df.rating.values
    test_user, test_item, test_rating = test_df.user_id.values, test_df.item_id.values, test_df.rating.values

    train_data = (train_user, train_item, train_rating)
    test_data = (test_user, test_item, test_rating)
    
    return train_data, test_data, num_users, num_items

def RMSELoss(x, xhat):
    return torch.sqrt(torch.mean((x-xhat)**2))
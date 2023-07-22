import os 
import pandas as pd
import torch

def load_dataset(args):
    ratings_df = pd.read_csv(os.path.join(args.dpath, 'rating.txt'), '\t',header=None, names=['item','user','rating','status',
                                                                'creation','last_modified','type','vertical_id'])
    social_df = pd.read_csv(os.path.join(args.dpath, 'user_rating.txt'), '\t', header=None, names=['user','target_user',
                                                                                                   'trust_value','creation'])
    
    ratings_df = ratings_df[['item','user','rating']]
    user2idx = {user:idx for idx, user in enumerate(ratings_df['user'].unique())}
    idx2user = {idx:user for idx, user in enumerate(ratings_df['user'].unique())}
    item2idx = {item:idx for idx, item in enumerate(ratings_df['item'].unique())}
    idx2item = {idx:item for idx, item in enumerate(ratings_df['item'].unique())}
    ratings_df['user'] = ratings_df['user'].map(user2idx)
    ratings_df['item'] = ratings_df['item'].map(item2idx)
    ratings_df['rating'] = ratings_df['rating'].apply(lambda x: (x-1)/4)
    
    ratings_df = ratings_df.loc[(ratings_df['user'] < args.user_num) & (ratings_df['item'] < args.item_num)]

    social_df = social_df[['user','target_user','trust_value']]
    social_df['user'] = social_df['user'].map(user2idx)
    social_df['target_user'] = social_df['target_user'].map(user2idx)
    social_df['trust_value'] = social_df['trust_value'].apply(lambda x: 1 if x > 0 else 0)
    social_df = social_df.loc[(social_df['user'] < args.user_num) & (social_df['target_user'] < args.user_num)]
    
    return ratings_df, social_df

def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2) + 1e-6)
import argparse 
import os 
import numpy as np 
import pandas as pd

import torch 
from torch.utils.data import DataLoader
from scipy.sparse import csr_matrix

from dataset import Netflix
from model import NeighborhoodModel, AsymmetricSVD, SVDPlusPlus, IntergratedModel
from utils import preprocess, RMSELoss, make_samples
from trainer import train, evaluate


def main(args):
    if args.preprocess == True:
        preprocess()
        
    df = pd.read_csv(args.dpath)
    
    train_df, test_df,  sample_df_user2idx, sample_df_item2idx = make_samples(df, args)
     
    train_dataset = Netflix(train_df)
    test_dataset = Netflix(test_df)
    
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    train_R = csr_matrix(
        (np.array(train_df['Rating'].values, dtype = np.int32),
        (np.array(train_df['Cust_ID'].values, dtype = np.int32),np.array(train_df['Movie_Id'].values, dtype = np.int32))
        ), shape = (len(sample_df_user2idx), len(sample_df_item2idx)))

    mu = train_df.Rating.mean() 
    F = args.F
    k = args.k 
    device = torch.device('cpu')
    
    if args.model_name == 'NeighborhoodModel':
        model = NeighborhoodModel(train_R, mu, k, device)
    elif args.model_name == 'AsymmetricSVD':
        model = AsymmetricSVD(train_R, mu, F, device)
    elif args.model_name == 'SVD++':
        model = SVDPlusPlus(train_R, mu, F, device)
    else:
        model = IntergratedModel(train_R, mu, F, k, device)
        
    model.to(device)
    summary = pd.DataFrame(columns=['model', 'epoch', 'train_rmse', 'test_rmse'])

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    criterion = RMSELoss
    early_stop_cnt = args.early_stop_cnt
    early_stop_loss = 100000

    for epoch in range(0,args.epochs):
        print(f'{args.model_name} model | {epoch} epoch start')
        train_loss = train(model, train_dataloader, criterion, optimizer, device)
        val_loss = evaluate(model, test_dataloader, criterion, device)
        
        print(f'{args.model_name} model epoch: {epoch}, train_loss: {train_loss}, val_loss: {val_loss}')
        summary = pd.concat([summary, pd.DataFrame([[args.model_name, epoch, train_loss, val_loss]], columns=['model', 'epoch', 'train_rmse', 'test_rmse'])])
        
        if early_stop_loss > val_loss:
            early_stop_cnt = 0
            early_stop_loss = val_loss
        else:
            early_stop_cnt += 1 
            
        if early_stop_loss < val_loss and early_stop_cnt > 10:
            break
        
    return summary 



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess', type=bool, default=False)
    parser.add_argument('--dpath', type=str, default='preprocessed_df_with_timestamp.csv')
    parser.add_argument('--user', type=int, default=10000)
    parser.add_argument('--item', type=int, default=388)
    
    parser.add_argument('--model_name', type=str, default='NeighborhoodModel')
    parser.add_argument('--F', type=int, default=15)
    parser.add_argument('--k', type=int, default=15)
    
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--early_stop_cnt', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100)
    
    args = parser.parse_args()
    
    summary = main(args)
    print(summary)
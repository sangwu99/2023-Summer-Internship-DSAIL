import argparse

import torch
import torch.nn as nn

import pandas as pd 

from utils import load_dataset
from dataset import load_and_split_dataset
from model import WideAndDeep
from trainer import train

def main(args):
    df = load_dataset(args)
    
    train_dataloader, test_dataloader, wide_dim, deep_dims = load_and_split_dataset(df, args)
    
    device = torch.device('cpu')
    model = WideAndDeep(wide_dim, deep_dims, args.factor_dim, args.layer_num, args.hidden_dim, args.output_dim)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)
    
    train(args, model, train_dataloader, test_dataloader, optimizer, criterion, device)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dpath', type=str, default='../ml-100k/')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    
    parser.add_argument('--lr', type=float, default=0.001)
    
    parser.add_argument('--factor_dim', type=int, default=16)
    parser.add_argument('--layer_num', type=int, default=3)
    parser.add_argument('--hidden_dim', type=list, default=[8, 4, 2])
    parser.add_argument('--output_dim', type=int, default=1)
    
    args = parser.parse_args()
    
    main(args)
    
    
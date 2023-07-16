import argparse 
import torch
import torch.nn as nn

from model import FM
from trainer import train
from dataset import load_data_split
from utils import load_dataset

def main(args):
    df = load_dataset(args.dpath)
    
    train_dataloaer, test_dataloader = load_data_split(args, df)
    
    input_dim = len(df.columns) - 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = FM(input_dim, args.factor_dim)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    train(args, model, train_dataloaer, criterion, optimizer, device)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dpath', type=str, default='../ml-100k/')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test_size', type=float, default=0.2)
    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    
    parser.add_argument('--factor_dim', type=int, default=10)
    
    args = parser.parse_args()
    
    main(args)
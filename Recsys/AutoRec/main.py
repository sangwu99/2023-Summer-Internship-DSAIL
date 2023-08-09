import argparse
import torch
from torch.utils.data import DataLoader

from utils import load_data, RMSELoss
from dataset import AutoRecDataset
from model import AutoRec
from trainer import train

def main(args):
    
    train_data, test_data, num_users, num_items = load_data(args)
    
    train_dataset = AutoRecDataset(train_data[0], train_data[1], train_data[2], 
                                   num_users, num_items, True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = AutoRecDataset(test_data[0], test_data[1], test_data[2], 
                                  num_users, num_items, True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoRec(num_users, num_items, args.hidden_dim, args.is_item)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.reg)
    criterion = RMSELoss
    
    train(args, model, train_loader, test_loader, criterion, optimizer, device)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dpath', type=str, default='../ml-100k')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--hidden_dim', type=list, default=[64])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--is_item', type=bool, default=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--reg', type=float, default=1e-4)
    
    args = parser.parse_args()
    
    main(args)
    
import argparse 
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from utils import load_dataset, RMSELoss
from dataset import Rating, Social
from model import SoRec
from trainer import train

def main(args):
    ratings_df, social_df = load_dataset(args)
        
    train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
    
    train_dataset, test_dataset = Rating(train_df), Rating(test_df)
    social_dataset = Social(social_df)
    
    rating_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    social_loader = DataLoader(social_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SoRec(args.user_num, args.item_num, args.num_factors).to(args.device)
    
    criterion = RMSELoss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    train(args, model, rating_loader, social_loader, test_loader, criterion, optimizer)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dpath', type=str, default='../epinions/')
    
    parser.add_argument('--user_num', type=int, default=5000)
    parser.add_argument('--item_num', type=int, default=5000)
    parser.add_argument('--num_factors', type=int, default=32)
    
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lambda_c', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=64)
    
    args = parser.parse_args()
    
    main(args)
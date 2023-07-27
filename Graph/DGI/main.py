import argparse 
import torch 
import torch.nn as nn 

from utils import load_dataset
from dataset import preprocess_adj_feature
from model import DGI
from trainer import train, test


def main(args):
    adj, feature, graph = load_dataset(args)
    adj, feature = preprocess_adj_feature(adj, feature)
    
    num_node, num_feature = graph.x.size()
    
    model = DGI(num_feature, args.hidden_dim)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    
    train(args, model, criterion, optimizer, adj, feature, num_node)
    
    model.load_state_dict(torch.load(f'best_{args.dataset_name}.pt'))
    
    embedding = model.get_embedding(feature, adj)
    embedding = embedding.squeeze(0).detach().numpy()
    label = graph.y.numpy()
    
    test(args, embedding, label)
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DGI')
    
    parser.add_argument('--dataset_name', type=str, default='Cora', help='Dataset name')
    
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test size ratio')
    
    args = parser.parse_args()
    
    main(args)
import argparse
import torch
from utils import load_graph
from dataset import split_dataset
from model import TransE
from trainer import train

def main(args):
    graph = load_graph()
    train_dataloader, val_dataloader, test_dataloader = split_dataset(graph, args.batch_size)
    
    args.num_entities = graph.num_nodes()
    args.num_labels = graph.num_edges()
    
    model = TransE(args)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MarginRankingLoss(margin=args.margin, reduction='none')

    train(args, model, train_dataloader, optimizer, criterion)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # model parameters
    parser.add_argument('--embedding_dim', type=int, default=50)
    
    # training parameters
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--neg_num', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    
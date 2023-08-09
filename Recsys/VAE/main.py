import argparse
import torch 

from dataset import get_dataloader
from model import VAE
from trainer import train


def main(args):
    torch.cuda.empty_cache()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataloader, test_dataloader = get_dataloader(args)
    model = VAE(args.input_dim, args.hidden_dim, args.latent_dim)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    train(args, model, train_dataloader, optimizer, device)
    
    
if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dpath', type=str, default='./data')
    parser.add_argument('--input_dim', type=int, default=784)
    parser.add_argument('--hidden_dim', type=list, default=[512, 256, 128])
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=128)
    
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=100)
    
    args = parser.parse_args()
    
    main(args)
    

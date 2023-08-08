import argparse 
import torch
import torch.nn as nn 
import torch.nn.functional as F

from model import Transformer
from trainer import train
from utils import preprocess_data
from dataset import load_dataloader

def main(args):
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    DOI, ENG, eng_list, doi_list, PAD_WORD = preprocess_data(args)
    train_iterator, src_vocab_size, trg_vocab_size, PAD_idx = load_dataloader(args, DOI, ENG, 
                                                                              eng_list, doi_list, PAD_WORD, device)
    
    model = model = Transformer(n_src_vocab = src_vocab_size, n_trg_vocab = trg_vocab_size, src_pad_idx = PAD_idx, trg_pad_idx = PAD_idx, 
                           d_word_vec=args.d_word_vec, d_model=args.d_model, d_inner=args.d_inner,
                           n_layers=args.n_layers, n_head=args.n_head, dropout=args.dropout, n_position=args.n_position,
                           trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True,
                           scale_emb_or_prj='prj').to(device)
                        
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = F.cross_entropy
    
    train(args, model, train_iterator, optimizer, device, criterion, PAD_idx)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer')
    
    parser.add_argument('--d_word_vec', type=int, default=512, help='Word embedding dimension')
    parser.add_argument('--d_model', type=int, default=512, help='Transformer model dimension')
    parser.add_argument('--d_inner', type=int, default=2048, help='Transformer ffn dimension')
    parser.add_argument('--n_layers', type=int, default=6, help='Number of layers')
    parser.add_argument('--n_head', type=int, default=8, help='Number of heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--n_position', type=int, default=200, help='Max position')
    
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--min_freq', type=int, default=3, help='Min freq')
    parser.add_argument('--epochs', type=int, default=100, help='Epochs')
    
    args = parser.parse_args()
    
    main(args)
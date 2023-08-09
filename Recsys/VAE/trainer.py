import torch 
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd 

def train(args, model, train_dataloader, optimizer, device):
    summary = pd.DataFrame(columns=['epoch', 'train_loss'])
    
    for epoch in range(args.epochs):
        train_loss = 0
        
        for i, (x, _) in enumerate(train_dataloader):
            # forward
            x = x.view(-1, args.input_dim)
            x = x.to(device)
            pred, mu, logvar = model(x)
            reconst_loss = F.binary_cross_entropy(pred, x, reduction='sum')
            kl_divergence = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            loss = reconst_loss + kl_divergence
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        train_loss /= len(train_dataloader)

        print(f'Epoch {epoch+1}, train loss: {train_loss:.4f}')
        summary = pd.concat([summary, pd.DataFrame([[epoch+1, train_loss]], columns=['epoch', 'train_loss'])], ignore_index=True)
        
    summary.to_csv('summary.csv', index=False)
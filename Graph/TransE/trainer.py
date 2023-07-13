import torch
import pandas as pd 


def train(args, model, dataloader, optimizer, criterion, device):
    summary = pd.DataFrame(columns=['Epoch', 'Loss'])
    model = model.to(device)
    
    for epoch in range(args.epochs):
        model.norm_entity()
        train_loss = 0
        print(f'Epoch {epoch}')
        
        for head, label, tail in dataloader:
            head, label, tail = head.repeat(args.neg_num).to(device) , label.repeat(args.neg_num).to(device), tail.repeat(args.neg_num).to(device)
            pos = model(head, tail, label).repeat(2)
            neg_head  = torch.randint(0, args.num_entities, (head.size(0),)).to(device)
            neg_tail = torch.randint(0, args.num_entities, (head.size(0),)).to(device)
            neg = torch.cat([model(neg_head, tail, label), model(head, neg_tail, label)], dim=0) 
            loss = criterion(pos, neg, -1 * torch.ones(2 * head.size(0)).to(device))
            
            optimizer.zero_grad()
            loss.mean().backward()
            train_loss += loss.mean().item()
            optimizer.step()
        
        train_loss /= len(dataloader)
            
        print(f'Epoch {epoch} | Loss: {train_loss}')
        
        summary = pd.concat([summary, pd.DataFrame([[epoch, train_loss]], columns=['Epoch', 'Loss'])])
    
    summary.to_csv('summary.csv', index=False)
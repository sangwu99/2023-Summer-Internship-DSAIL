import pandas as pd 
import torch


def train(args, model, train_loader, test_loader, criterion, optimizer, device):
    summary = pd.DataFrame(columns=['epoch', 'train_loss', 'test_loss'])
    
    for epoch in range(args.epochs):
        model.train()
        
        for batch in train_loader:
            batch = batch.to(device)
            mask = batch >= 0 
            neg = batch == -1 
            batch[neg] = 0.5 
            
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output[mask], batch[mask])
            
            loss.backward()
            optimizer.step()
            
            train_loss = loss.item()
            
        train_loss = train_loss / len(train_loader)
        
        with torch.no_grad():
            model.eval()
            
            for batch in test_loader:
                batch = batch.to(device)
                mask = batch >= 0 
                
                output = model(batch)
                loss = criterion(output[mask], batch[mask])
                
                test_loss = loss.item()
                
            test_loss = test_loss / len(test_loader)
            
        summary = pd.concat([summary, pd.DataFrame([[epoch, train_loss, test_loss]], columns=['epoch', 'train_loss', 'test_loss'])])
        
    summary.to_csv('summary.csv', index=False)
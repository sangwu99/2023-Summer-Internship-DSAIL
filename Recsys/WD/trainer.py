import torch
import pandas as pd

def train(args, model, train_dataloader, test_dataloader, optimizer, criterion, device):
    summary = pd.DataFrame(columns=['Epoch', 'Loss', 'Test_Loss'])

    for epoch in range(args.epochs):
        
        print(f'Epoch {epoch}')

        model.train()
        train_loss = 0.0
        for wide, deep, y in train_dataloader:
            wide, deep, y = wide.to(device), deep.to(device), y.to(device)
            optimizer.zero_grad()
            
            pred = model(wide, deep)
            loss = criterion(pred.squeeze(), y.to(torch.float32))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_dataloader)
        
        model.eval()
        
        test_loss = 0.0
        for wide, deep, y in test_dataloader:
            with torch.no_grad():
                wide, deep, y = wide.to(device), deep.to(device), y.to(device)
                pred = model(wide, deep)
                loss = criterion(pred.squeeze(), y.to(torch.float32))
                test_loss += loss.item()
            
        test_loss /= len(test_dataloader)
        
        print(f'Epoch {epoch} | Loss: {train_loss} | Test Loss: {test_loss}')
        
        summary = pd.concat([summary, pd.DataFrame([[epoch, train_loss, test_loss]], columns=['Epoch', 'Loss', 'Test_Loss'])])

    summary.to_csv('summary.csv', index=False) 
    
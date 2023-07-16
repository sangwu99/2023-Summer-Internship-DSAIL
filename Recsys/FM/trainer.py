import pandas as pd 


def train(args, model, dataloader, criterion, optimizer, device):
    
    summary = pd.DataFrame(columns=['Epoch', 'Loss'])

    for epoch in range(args.epochs):
        
        print(f'Epoch {epoch}')

        model.train()
        train_loss = 0.0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(dataloader)
        
        print(f'Epoch {epoch} | Loss: {train_loss}')
        
        summary = pd.concat([summary, pd.DataFrame([[epoch, train_loss]], columns=['Epoch', 'Loss'])])
    
    summary.to_csv('summary.csv', index=False)    
        

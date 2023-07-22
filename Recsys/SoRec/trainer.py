import torch
import pandas as pd 


def train(args, model, rating_loader, social_loader, test_loader, criterion, optimizer):
    summary = pd.DataFrame(columns=['epoch', 'rating_loss', 'social_loss', 'test_loss'])
    
    for epoch in range(args.epochs):
        model.train()
        
        rating_loss = 0
        social_loss = 0
        test_loss = 0

        print(f'Epoch {epoch+1}')
        
        for user, item, rating in rating_loader:
            user, item, rating = user.to(args.device), item.to(args.device), rating.to(args.device)
            pred = model.rating(user, item)
            loss = criterion(pred, rating)
            optimizer.zero_grad()
            
            loss.backward()
            optimizer.step()
            
            rating_loss += loss.item()
        
        rating_loss = rating_loss / len(rating_loader)
        
        
        for user, target_user, trust_value in social_loader:
            user, target_user, trust_value = user.to(args.device), target_user.to(args.device), trust_value.to(args.device)
            pred = model.social(user, target_user)
            loss = criterion(pred, trust_value)
            loss = loss * args.lambda_c
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            
            social_loss += loss.item()
            
        social_loss = social_loss / len(social_loader)

        
        with torch.no_grad():
            for user, item, rating in test_loader:
                user, item, rating = user.to(args.device), item.to(args.device), rating.to(args.device)
                pred = model.rating(user, item)
                loss = criterion(pred, rating)
                
                test_loss += loss.item()
                
        test_loss = test_loss / len(test_loader)

        print(f'Epoch: {epoch}, Rating Loss: {rating_loss:.4f}, Social Loss: {social_loss:.4f}, Test Loss: {test_loss:.4f}')
        
        summary = pd.concat([summary, pd.DataFrame([[epoch, rating_loss, social_loss, test_loss]], columns=['epoch', 'rating_loss', 'social_loss', 'test_loss'])])
        
    summary.to_csv('summary.csv', index=False)
            
import torch 

def train(model, train_loader, criterion ,optimizer, device):
    model.train() 
    total_loss = 0 
    for user, item, rating in train_loader:
        user = user.to(device)
        item = item.to(device)
        rating = rating.to(device)
        
        optimizer.zero_grad()
        pred = model(user, item)
        loss = criterion(pred, rating)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for user, item, rating in test_loader:
            try:
                user = user.to(device)
                item = item.to(device)
                rating = rating.to(device)

                pred = model(user, item)
                loss = criterion(pred, rating)

                total_loss += loss.item()
            except:
                pass
    
    return total_loss / len(test_loader)


def temporal_train(model, train_loader, criterion, optimizer, device):
    model.train() 
    total_loss = 0 
    for user, item, time,rating in train_loader:
        user = user.to(device)
        item = item.to(device)
        time = time.to(device)
        rating = rating.to(device)
        
        optimizer.zero_grad()
        pred = model(user, item,time)
        loss = criterion(pred, rating)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    total_loss = total_loss / len(train_loader)
    
    return total_loss



def temporal_evaluate(model, test_loader, criterion, optimizer, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for user, item, time,rating in test_loader:
            user = user.to(device)
            item = item.to(device)
            time = time.to(device)
            rating = rating.to(device)

            pred = model(user, item,time)
            loss = criterion(pred, rating)

            total_loss += loss.item()
    
    return total_loss / len(test_loader)
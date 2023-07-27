import torch
import numpy as np
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

def train(args, model, criterion, optimizer, adj, feature, num_node):
    summary = pd.DataFrame(columns=['Epoch', 'Loss'])
    best_loss = 1000000

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        
        pos_lb = torch.ones(args.batch_size, num_node)
        neg_lb = torch.zeros(args.batch_size, num_node)
        pos_neg_lb = torch.cat((pos_lb, neg_lb), dim=1)
        
        corrupted_node = np.random.permutation(num_node)
        corrupted_features = feature[:, corrupted_node, :]
        logit = model(feature, corrupted_features, adj) 
        loss = criterion(logit, pos_neg_lb)
        
        if best_loss > loss.item():
            best_loss = loss.item()
            torch.save(model.state_dict(), f'best_{args.dataset_name}.pt')
        
        loss.backward() 
        optimizer.step()
        
        print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch, loss.item()))
        
        summary = pd.concat([summary, pd.DataFrame([[epoch, loss.item()]], columns=['Epoch', 'Loss'])], ignore_index=True)
        
    summary.to_csv(f'{args.dataset_name}_summary.csv', index=False)
        
        
def test(args, embedding, label):
    result = pd.DataFrame(columns=['Dataset', 'Accuracy', 'Macro-F1', 'Micro-F1'])
    train_x, test_x, train_y, test_y = train_test_split(embedding, label, test_size=args.test_size, random_state=42)
    
    LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    LR.fit(train_x, train_y)
    
    pred = LR.predict(test_x)
    acc = accuracy_score(test_y, pred)
    macro_f1 = f1_score(test_y, pred, average='macro')
    micro_f1 = f1_score(test_y, pred, average='micro')
    
    print('Accuracy: {:.4f}, Macro-F1: {:.4f}, Micro-F1: {:.4f}'.format(acc, macro_f1, micro_f1))
    result = pd.concat([result, pd.DataFrame([[args.dataset_name, acc, macro_f1, micro_f1]], columns=['Dataset', 'Accuracy', 'Macro-F1', 'Micro-F1'])], ignore_index=True)
    
    result.to_csv(f'{args.dataset_name}_result.csv', index=False)
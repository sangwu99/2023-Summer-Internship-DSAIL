import torch


def train(args, model, dataloader, optimizer, criterion, device):
    
    for epoch in range(args.epochs):
        model.norm_entity()
        train_loss = 0
        
        for head, label, tail in dataloader:
            head, label, tail = head.repeat(args.neg_num) , label.repeat(args.neg_num), tail.repeat(args.neg_num)
            pos = model(head, tail, label).repeat(2)
            neg_head  = torch.randint(0, args.num_entities, (args.batch_size * args.neg_num,))
            neg_tail = torch.randint(0, args.num_entities, (args.batch_size * args.neg_num,))
            neg = torch.cat([model(neg_head, tail, label), model(head, neg_tail, label)], dim=0) 
            loss = criterion(pos, neg, -1 * torch.ones(2 * args.batch_size * args.neg_num))
            
            optimizer.zero_grad()
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            
        print(f'Epoch {epoch} | Loss: {train_loss}')
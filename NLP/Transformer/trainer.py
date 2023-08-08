from tqdm import tqdm
import pandas as pd 

def patch_src(src, pad_idx):
    src = src.transpose(0, 1)
    return src

def patch_trg(trg, pad_idx):
    trg = trg.transpose(0, 1)
    trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
    return trg, gold

def train(args, model, train_iterator, optimizer, device, criterion, pad_idx):
    summary = pd.DataFrame(columns=['epoch', 'train_loss'])

    train_loss = [] 

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0 

        for batch in tqdm(train_iterator):

            # prepare data
            src_seq = patch_src(batch.src, pad_idx).to(device)
            trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch.trg, pad_idx))

            # forward
            optimizer.zero_grad()
            pred = model(src_seq, trg_seq)

            # backward and update parameters
            loss = criterion(pred, gold, ignore_index = pad_idx, reduction='sum')
            loss.backward()
            optimizer.step()

            # note keeping
            total_loss += loss.item()
        
        train_loss.append(total_loss / len(train_iterator))
        print(f'Epoch [{epoch+1}/{100}] loss: {total_loss / len(train_iterator):.3f}')
        summary = pd.concat([summary, pd.DataFrame([[epoch+1, total_loss / len(train_iterator)]], columns=['epoch', 'train_loss'])], ignore_index=True)
        
    summary.to_csv('summary.csv', index=False)
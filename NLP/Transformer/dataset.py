import torchtext
from torchtext.data import Field, Dataset, BucketIterator 

import easydict


def load_dataloader(args, DOI, ENG, eng_list, doi_list, PAD_WORD, device):
    train = []
    for i,j in zip(eng_list, doi_list):
        temp_dict = easydict.EasyDict({'src': j, 'trg': i})
        train.append(temp_dict)
        
    PAD_idx = DOI.vocab.stoi[PAD_WORD]
    src_vocab_size = len(DOI.vocab)
    trg_vocab_size = len(ENG.vocab)
    
    fields = {'src': DOI, 'trg': ENG}

    train = Dataset(examples=train, fields=fields)

    train_iterator = BucketIterator(train, batch_size=args.batch_size, device=device, train=True)
    
    return train_iterator, src_vocab_size, trg_vocab_size, PAD_idx
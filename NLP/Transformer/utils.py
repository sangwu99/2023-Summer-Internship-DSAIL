from datasets import load_dataset 
import spacy
import torchtext

def preprocess_data(args):
    dataset = load_dataset('wmt16', 'de-en', split='train[:1%]')
    
    src_lang_model = spacy.load('de_core_news_sm') 
    trg_lang_model = spacy.load('en_core_web_sm')
    
    PAD_WORD = '<blank>' 
    UNK_WORD = '<unk>'
    BOS_WORD = '<s>' 
    EOS_WORD = '</s>' 
    
    def tokenize_src(text):
        return [tok.text for tok in src_lang_model.tokenizer(text)]

    def tokenize_trg(text):
        return [tok.text for tok in trg_lang_model.tokenizer(text)]

    ENG = torchtext.data.Field(
        tokenize=tokenize_src, lower=False,
        pad_token=PAD_WORD, init_token=BOS_WORD, eos_token=EOS_WORD)

    DOI = torchtext.data.Field(
        tokenize=tokenize_trg, lower=False,
        pad_token=PAD_WORD, init_token=BOS_WORD, eos_token=EOS_WORD)
    
    eng_list = []
    doi_list = []
    for i in dataset['translation']:
        eng_list.append(i['en'].split(' '))
        doi_list.append(i['de'].split(' '))
        
    ENG.build_vocab(eng_list, min_freq=args.min_freq)
    DOI.build_vocab(doi_list, min_freq=args.min_freq)
    
    for w, _ in DOI.vocab.stoi.items():
        if w not in ENG.vocab.stoi:
            ENG.vocab.stoi[w] = len(ENG.vocab.stoi)
    ENG.vocab.itos = [None] * len(ENG.vocab.stoi)
    for w, i in ENG.vocab.stoi.items():
        ENG.vocab.itos[i] = w
    DOI.vocab.stoi = ENG.vocab.stoi
    DOI.vocab.itos = ENG.vocab.itos
    
    return DOI, ENG, eng_list, doi_list, PAD_WORD
    
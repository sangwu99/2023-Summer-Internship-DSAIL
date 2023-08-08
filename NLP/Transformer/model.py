import torch 
import torch.nn as nn
import torch.nn.functional as F

import numpy as np 

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_model // n_head

        self.WQ = nn.Linear(d_model, d_model, bias=False)
        self.WK = nn.Linear(d_model, d_model, bias=False)
        self.WV = nn.Linear(d_model, d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model, bias=False)
        
    def ScaleDotProductAttention(self, q, k, v, mask=None):
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)  

        if mask is not None:
            mask = mask.unsqueeze(1)  
            attn_scores = attn_scores.masked_fill_(mask == False, -1 * 1e12)

        attn_dists = F.softmax(attn_scores, dim=-1) 

        attn_values = torch.matmul(attn_dists, v)  

        return attn_values, attn_dists

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]

        q = self.WQ(q)  
        k = self.WK(k)  
        v = self.WV(v)  

        q = q.view(batch_size, -1, self.n_head, self.d_k)
        k = k.view(batch_size, -1, self.n_head, self.d_k)
        v = v.view(batch_size, -1, self.n_head, self.d_k) 

        q = q.transpose(1, 2) 
        k = k.transpose(1, 2) 
        v = v.transpose(1, 2)  

        attn_values, attn_dists = self.ScaleDotProductAttention(q, k, v, mask=mask) 
        attn_values = attn_values.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model) 

        return self.fc(attn_values), attn_dists
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid)) 
        
    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
        
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) 
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) 
        
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)
    
    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach() 
    
class FeedForwardNetwork(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, in_features)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(in_features)
        
    def forward(self, x):
        res = x 
        x = self.linear2(F.relu(self.linear1(x)))
        x = self.dropout(x)
        x += res 
        
        x = self.layer_norm(x)
        
        return x
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ffn, dropout):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_head)
        self.ffn = FeedForwardNetwork(d_model, d_ffn, dropout=dropout)

    def forward(self, enc_input, mask=None):
        enc_output, attention = self.attention(enc_input, enc_input, enc_input, mask=mask)
        enc_output = self.ffn(enc_output)
        return enc_output, attention

class Encoder(nn.Module):
    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, 
            d_model, d_ffn, pad_idx, dropout=0.1, n_position=200, scale_emb=False):

        super().__init__()
        
        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, n_head, d_ffn, dropout=dropout)
            for _ in range(n_layers)])
        
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=True):

        enc_slf_attn_list = []

        enc_output = self.src_word_emb(src_seq)
        if self.scale_emb:
            enc_output *= self.d_model ** 0.5
            
        enc_output = self.dropout(self.position_enc(enc_output))
        
        enc_output = self.layer_norm(enc_output)
        
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []
            
        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,
    
class DecoderLayer(nn.Module):

    def __init__(self, d_model, d_ffn, n_head, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        self.slf_attn = MultiHeadAttention(d_model, n_head)
        self.enc_attn = MultiHeadAttention(d_model, n_head)
        self.pos_ffn = FeedForwardNetwork(d_model, d_ffn, dropout=dropout)

    def forward(
            self, dec_input, enc_output,
            slf_attn_mask=None, dec_enc_attn_mask=None):
        
        dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        
        dec_output = self.pos_ffn(dec_output)
        
        return dec_output, dec_slf_attn, dec_enc_attn
    
class Decoder(nn.Module):
    def __init__(
            self, n_trg_vocab, d_word_vec, n_layers, n_head,
            d_model, d_ffn, pad_idx, n_position=200, dropout=0.1, scale_emb=False):

        super().__init__()
        
        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_ffn, n_head, dropout=dropout)
            for _ in range(n_layers)])
        
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=True):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        dec_output = self.trg_word_emb(trg_seq)
        if self.scale_emb:
            dec_output *= self.d_model ** 0.5
            
        dec_output = self.dropout(self.position_enc(dec_output))
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,
    
def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class Transformer(nn.Module):
    def __init__(
            self, n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, dropout=0.1, n_position=200,
            trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True,
            scale_emb_or_prj='prj'):

        super().__init__()
        
        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx

        assert scale_emb_or_prj in ['emb', 'prj', 'none']
        scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False
        self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False
        self.d_model = d_model

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_ffn=d_inner,
            n_layers=n_layers, n_head=n_head,
            pad_idx=src_pad_idx, dropout=dropout, scale_emb=scale_emb)

        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_ffn=d_inner,
            n_layers=n_layers, n_head=n_head, 
            pad_idx=trg_pad_idx, dropout=dropout, scale_emb=scale_emb)

        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        assert d_model == d_word_vec

        if trg_emb_prj_weight_sharing:
            self.trg_word_prj.weight = self.decoder.trg_word_emb.weight

        if emb_src_trg_weight_sharing:
            self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight

    def forward(self, src_seq, trg_seq):
        
        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

        enc_output, *_ = self.encoder(src_seq, src_mask)
        
        dec_output, attention1, attention2 = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
        
        seq_logit = self.trg_word_prj(dec_output)
        if self.scale_prj:
            seq_logit *= self.d_model ** -0.5

        return seq_logit.view(-1, seq_logit.size(2))
            
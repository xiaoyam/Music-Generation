import sys
import math
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import load_config, print_config, load_drum
class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        
        # perform linear operation and split into h heads
        
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * d_model
       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
# calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        
        output = self.out(concat)
    
        return output

    def freeze_weights(self):
        for p in self.q_linear.parameters():
            p.requires_grad = False
        for p in self.v_linear.parameters():
            p.requires_grad = False
        for p in self.k_linear.parameters():
            p.requires_grad = False
        for p in self.out.parameters():
            p.requires_grad = False

def attention(q, k, v, d_k, mask=None, dropout=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    # if mask is not None:
    #         mask = mask.unsqueeze(1)
    #         scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)
        
    if dropout is not None:
        scores = dropout(scores)
            
    output = torch.matmul(scores, v)
    return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__() 
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
    def freeze_weights(self):
        for p in self.linear_1.parameters():
            p.requires_grad = False
        for p in self.linear_2.parameters():
            p.requires_grad = False

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm
    def freeze_weights(self):
        self.alpha.requires_grad = False
        self.bias.requires_grad = False

# build an encoder layer with one multi-head attention layer and one # feed-forward layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout = 0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x
    
# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, d_model)
        self.attn_2 = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)

    def forward(self, x, e_outputs, src_mask, trg_mask):
            x2 = self.norm_1(x)
            x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
            x2 = self.norm_2(x)
            x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs,
            src_mask))
            x2 = self.norm_3(x)
            x = x + self.dropout_3(self.ff(x2))
            return x
    def freeze_hiddens(self):
        self.norm_1.freeze_weights()
        self.norm_2.freeze_weights()
        self.norm_3.freeze_weights()
        self.attn_1.freeze_weights()
        self.attn_2.freeze_weights()
        self.ff.freeze_weights()

    # We can then build a convenient cloning function that can generate multiple layers:
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads):
        super().__init__()
        self.N = N
        # self.embed = Embedder(vocab_size, d_model)
        # self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)
    def forward(self, src, mask):
        x = src
        # x = self.embed(src)
        # x = self.pe(x)
        for i in range(N):
            x = self.layers[i](x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, d_model, N, heads):
        super().__init__()
        self.N = N
        self.layers = get_clones(DecoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)
    def forward(self, latents, e_outputs, src_mask, trg_mask):
        x = latents
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)
    def freeze_hiddens(self):
        for l in self.layers:
            l.freeze_hiddens()
        self.norm.freeze_weights()

class Transformer(nn.Module):
    def __init__(self, spect_size, d_model, N, heads, activa_fn):
        super().__init__()
        #self.encoder = Encoder(src_vocab, d_model, N, heads)
        self.decoder = Decoder(d_model, N, heads)
        self.out_spect = nn.Linear(d_model, spect_size)
        self.out_mask = nn.Linear(d_model, spect_size)
        #self.out.append(self.out_spect)
        self.activation = lambda x: F.sigmoid(x)
    def forward(self, latents, src_mask=None, trg_mask=None):
        d_output = self.decoder(latents, latents, src_mask, trg_mask)
        output = self.out_spect(d_output)    
        mask = self.out_mask(d_output)
        mask_activa = self.activation(mask)
        return output, mask_activa
    def freeze_hiddens(self):
        self.decoder.freeze_hiddens()
        for p in self.out_spect.parameters():
            p.requires_grad = False
        for p2 in self.out_mask.parameters():
            p2.requires_grad = False
        
         
# we don't perform softmax on the output as this will be handled 
# automatically by our loss function


def train_model(epochs, train_y, print_every=100):
    
    model.train()
    
    start = time.time()
    temp = start
    batch_size = 32
    n_points = train_y.size()[0]
    n_batches = n_points // batch_size
    total_loss = 0
    order = np.random.permutation(n_points)

    
    for epoch in range(epochs):
       
        for i in range(n_batches):

            idxes = order[i * batch_size: (i + 1) * batch_size]


            src = train_y[idxes]
            trg = train_y[idxes]
            # the French sentence we input has all words except
            # the last, as it is using each word to predict the next
            
            # trg_input = trg[:, :-1]
            trg_input = trg
            
            # the words we are trying to predict
            
            # targets = trg[:, 1:].contiguous().view(-1)
            targets = trg_input
            
            # create function to make masks using mask code above
            
            # src_mask, trg_mask = create_masks(src, trg_input)
            src_mask = None
            trg_mask = None
            
            preds = model(src, trg_input, src_mask, trg_mask)
            
            optim.zero_grad()
            
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)),
            targets, ignore_index=target_pad)
            loss.backward()
            optim.step()
            
            total_loss += loss.data[0]
            if (i + 1) % print_every == 0:
                loss_avg = total_loss / print_every
                print("time = %dm, epoch %d, iter = %d, loss = %.3f,\
                %ds per %d iters" % ((time.time() - start) // 60,
                epoch + 1, i + 1, loss_avg, time.time() - temp,
                print_every))
                total_loss = 0
                temp = time.time()

def test_model(model, src):
    model.eval()
    src_mask = None
    trg_input = src
    preds = model(src, trg_input)
    h5f = h5py.File('preds.h5', 'w')
    h5f.create_dataset('preds', data=preds)
    h5f.close()




def main():
    args = []
    fname_start = '../data/drum/drum'
    data = load_drum(args, fname_start)
    (train_y, train_mask), (test_y, test_mask), test_y_true = data

    d_model = 512
    heads = 8
    N = 6
    src_vocab = len(train_y)
    trg_vocab = len(train_y)
    model = Transformer(src_vocab, trg_vocab, d_model, N, heads)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    # this code is very important! It initialises the parameters with a
    # range of values that stops the signal fading or getting too big.
    # See this blog for a mathematical explanation.
    optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)


    train_model(1000, train_y)
    test_model(model, test_y)


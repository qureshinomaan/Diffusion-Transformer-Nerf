import torch 
import torch.nn as nn 
import math

class InputEmbeddings(nn.Module):
    """ Input Embeddings take the input sequence and convert it into a vector of size d_model

    Args:
        d_model (int): The size of the output vector
        vocab_size (int): The size of the 
        
    Output:
        x (torch.Tensor): The output tensor of shape (batch_size, seq_len, d_model)
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)


    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    


class PositionalEncoding(nn.Module):
    """Positional encoding is a way of adding information about the relative position of the tokens in the sequence.

    Args:
        d_model (int) : Size of embedding of each word. 
        seq_len (int) : Length of the sequence or sentence or number of words in the sentence
        dropout (float) : The dropout probability
    """

    def __init__(self, d_model:int, seq_len: int, dropout: float):
        super().__init__()

        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout()


        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # Create a vector shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))

        # Apply the sin to even positions 
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) #(1, seq_len, d_model)

        self.register_buffer('pe', pe)


    def forward(self, x):

        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)

        return self.dropout(x)
    


class LayerNormalization(nn.Module):
    
    def __init__(self, eps:float=10e-6) -> None:
        super().__init__()
        self.eps = eps

        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)

        std = x.std(dim=-1, keepdim=True) ## keepdim mean keep the dimension along with means is applied, just convert it to 1


        return self.alpha * (x-mean)/(std + self.eps) + self.bias
    

class FeedForwardBlock(nn.Module):
    
    def __init__(self, d_model:int, d_ff:int, dropout:float) -> None:
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)


    def forward(self, x):
        self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model:int, h:int, dropout: float) -> None:
        super().__init__()

        self.d_model = d_model
        self.h = h

        assert d_model%h == 0, "d_model should be divisible by h"

        self.d_k = d_model // h 

        self.W_q = nn.Linear(d_model, d_model) #Wq
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)
        self.droptout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout : nn.Dropout):
        d_k = query.shape[-1] 
        
        attention_scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.maksed_fill(mask==0, -1e9)

        attention_scores = attention_scores.softmax(dim = -1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value), attention_scores


    def forward(self, q, k, v, mask):
        query = self.W_q(q) #(Batchsize, seqlen, d_model) -> (Batchsize, seqlen, dmodel)
        key = self.W_K(k) #Batchsize, seqlen, d_model) -> (Batchsize, seqlen, dmodel)
        value = self.W_v(v) #Batchsize, seqlen, d_model) -> (Batchsize, seqlen, dmodel)

        batch_size = query.shape[0]
        seq_len = query.shape[1]

        # Split the query, key and value into h heads
        query = query.view(batch_size, seq_len, self.h, self.d_k)
        key = key.view(batch_size, seq_len, self.h, self.d_k)
        value = value.view(batch_size, seq_len, self.h, self.d_k)

        
        #change the shape to (batch_size, self.h, seq_len, self.d_k)
        query = torch.permute(query, (0, 2, 1, 3))  ##(batch_size, self.h, seq_len, self.d_k)
        key = key.permute(query, (0, 2, 1, 3))      ## (batch_size, self.h, seq_len, self.d_k)
        value = value.permute(query, (0, 2, 1, 3))  ## (batch_size, self.h, seq_len, self.d_k)

        
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        x = x.transose(1,2)
        x = x.view(batch_size, seq_len, self.h*self.d_k)

        x = self.W_o(x)

        return x    

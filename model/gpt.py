import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    def __init__(self, embedding_dim, head_dim, block_size, dropout_rate):
        super().__init__()
        self.key = nn.Linear(embedding_dim, head_dim, bias=False)
        self.query = nn.Linear(embedding_dim, head_dim, bias=False)
        self.value = nn.Linear(embedding_dim, head_dim, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        # compute attention scores
        weight = torch.matmul(k, q.permute(0, 2, 1)) / C**2 # (B, T, C) @ (B, C, T) -> (B, T, T)
        weight = weight.masked_fill(self.tril[: T, : T] == 0, float('-inf')) # (B, T, T)
        weight = F.softmax(weight, dim=-1) # (B, T, T)
        weight = self.dropout(weight) # (B, T, T)
        out = torch.matmul(weight, v) # (B, T, T) @ (B, T, C) -> (B, T, C)
        
        return out
    

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, embedding_dim, head_dim, block_size, dropout_rate):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(embedding_dim, head_dim, block_size, dropout_rate) for _ in range(num_heads)]
        )
        self.projection = nn.Linear()
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.projection(out)
        out = self.dropout(out)
        
        return out
        
        
class FeedFoward(nn.Module):
    def __init__(self, embedding_dim, dropout_rate):
        super().__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Dropout(dropout_rate)
        )
        
    def forward(self, x):
        out = self.feed_forward(x)
        
        return out
    
    
class Block(nn.Module):
    def __init__(self, num_heads, embedding_dim, block_size, dropout_rate):
        super().__init__()
        head_dim = embedding_dim // num_heads
        self.multi_head_attention = MultiHeadAttention(num_heads, embedding_dim, head_dim, block_size, dropout_rate)
        self.ffwd = FeedFoward(embedding_dim, dropout_rate)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        x = x + self.multi_head_attention(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        
        return x
    
    
class GPT(nn.Module):
    def __init__(self, vocal_size, embedding_dim, block_size, num_blocks, num_heads, dropout_rate):
        super().__init__()
        self.block_size = block_size
        
        self.token_embedding = nn.Embedding(vocal_size, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        self.blocks = nn.Sequential(
            *[Block(num_heads, embedding_dim, block_size, dropout_rate) for _ in range(num_blocks)]
        )
        self.ln = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocal_size)
        
    def forward(self, idx):
        B, T = idx.shape
        token_embed = self.token_embedding(idx) # (B, T) -> (B, T, C)
        pos_embed = self.position_embedding(torch.range(T)) # (T, C)
        embed = token_embed + pos_embed # (B, T, C) + (T, C) -> (B, T, C)
        x = self.blocks(embed)
        x = self.ln(x)
        logits = self.lm_head(x)
        
        return logits
    
    def generate(self, idx, max_len):
        for _ in range(max_len):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:] # (B, T, C)
            logits = self(idx_cond) # (B, T, C)
            # get last time step 
            logits = logits[:, -1, :] # (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=-1) # (B, T + 1)
        
        return idx
        
        
        
            

if __name__ == '__main__':
    embedding_dim = 512
    num_heads = 1
    head_dim = embedding_dim // num_heads
    block_size = 32
    dropout_rate = 0.2
    
    # head = Head(embedding_dim, head_dim, block_size, dropout_rate)
    # x = torch.randn(4, 32, 512)
    # y = head(x)
    # print(y.shape)
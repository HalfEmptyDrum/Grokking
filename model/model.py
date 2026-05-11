import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import einsum

class MLP(nn.Module):
    def __init__(self, d_model=128, d_mlp=512):
        super().__init__()
        self.fc_in = nn.Linear(d_model, d_mlp)
        self.act = nn.ReLU()
        self.fc_out = nn.Linear(d_mlp, d_model)

    def forward(self, x):
        return self.fc_out(self.act(self.fc_in(x)))


class GrokModel(nn.Module):
    def __init__(self, prime=113, d_vocab=114, d_model=128,
                 num_heads=4, d_mlp=512, n_ctx=3):
        super().__init__()
        self.embedding = nn.Embedding(d_vocab, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(n_ctx, d_model) / d_model**0.5)
        self.attn = nn.MultiheadAttention(embed_dim=d_model,
                                          num_heads=num_heads,
                                          batch_first=True)
        self.mlp = MLP(d_model, d_mlp)
        self.unembedding = nn.Linear(d_model, prime)

    def forward(self, x):
        # x: [batch, 3] of token IDs
        x = self.embedding(x) + self.pos_embedding   # [batch, 3, d_model]
        attn_out, _ = self.attn(x, x, x)             # discard attention weights
        x = x + attn_out                             # residual after attention
        x = x + self.mlp(x)                          # residual after MLP
        logits = self.unembedding(x)                 # [batch, 3, prime]
        return logits
    
class SimpleModel(nn.Module):
    def __init__(self, p=113, d_hidden=100):
        super().__init__()
        self.p = p
        self.fc1 = nn.Linear(2 * p, d_hidden)
        self.fc2 = nn.Linear(d_hidden, p)

    def forward(self, a, b):
        x_a = F.one_hot(a, num_classes=self.p).float()
        x_b = F.one_hot(b, num_classes=self.p).float()
        x = torch.cat([x_a, x_b], dim=-1)
        h = F.relu(self.fc1(x))
        return self.fc2(h)
        
        
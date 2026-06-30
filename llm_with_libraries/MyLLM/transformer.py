import torch
import torch.nn as nn
from torch.nn import functional as F
from attention import CausalSelfAttention
from feedforward import FeedForward

class LayerNorm(nn.Module):
    """ LayerNorm with optional bias support (useful for bias=False architectures). """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class Block(nn.Module):
    """ A standard pre-LN Transformer decoder block. """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = FeedForward(config)

    def forward(self, x):
        # Pre-LN structure: residual connection after normalization
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

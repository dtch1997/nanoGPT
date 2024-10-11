import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from jaxtyping import Float

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, 
        ndim: int,
        bias: bool = True,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input: Float[torch.Tensor, "... ndim"]) -> Float[torch.Tensor, "... ndim"]:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    """ Multi-head attention module with causal mask """

    n_head: int
    d_model: int
    dropout: float
    block_size: int # Maximum sequence length. Needed for initializing causal mask
    bias: bool

    def __init__(self, 
        n_head: int,
        d_model: int,
        dropout: float,
        block_size: int,
        bias: bool = True
    ):             
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.dropout = dropout
        self.block_size = block_size
        self.bias = bias
        self._setup_modules()

    def _setup_modules(self):
        assert self.d_model % self.n_head == 0, "n_head must divide d_model"
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.d_model, 3 * self.d_model, bias=self.bias)
        # output projection
        self.c_proj = nn.Linear(self.d_model, self.d_model, bias=self.bias)
        # regularization
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            mask = torch.tril(torch.ones(self.block_size, self.block_size))
            mask = mask.view(1, 1, self.block_size, self.block_size)
            self.register_buffer("causal_attn_mask", mask)

    def forward(self, x: Float[torch.Tensor, "batch seq d_model"]) -> Float[torch.Tensor, "batch seq d_model"]:
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (d_model)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.d_model, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.causal_attn_mask[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    """ Multi-layer perceptron module with 4x hidden dim expansion """

    def __init__(self,         
        d_model: int,
        dropout: float
    ):
        super().__init__()
        self.c_fc    = nn.Linear(d_model, 4 * d_model, bias=True)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear( 4 * d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
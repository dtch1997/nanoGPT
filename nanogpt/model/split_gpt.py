import math
import inspect
import einops
import torch 
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from jaxtyping import Float, Int

from nanogpt.nn import LayerNorm, CausalSelfAttention, MLP
from nanogpt.model.model import Model

def split_resid_into_read_and_write(
    x: Float[torch.Tensor, "... d_resid"], 
    d_resid_read: int,
    d_resid_write: int
) -> tuple[Float[torch.Tensor, "... d_resid_read"], Float[torch.Tensor, "... d_resid_write"]]:
    """ Split the residual stream into the read and write streams """
    d_resid = x.shape[-1]
    assert d_resid_read + d_resid_write == d_resid, f"Expected d_resid_read + d_resid_write = d_resid; got {d_resid_read} + {d_resid_write} != {d_resid}"
    x_read, x_write = x.tensor_split([d_resid_read], dim=-1)
    return x_read, x_write

def merge_resid_read_and_write(
    x_read: Float[torch.Tensor, "... d_resid_read"],
    x_write: Float[torch.Tensor, "... d_resid_write"]
) -> Float[torch.Tensor, "... d_resid"]:
    """ Merge the read and write streams into the residual stream """
    return torch.cat([x_read, x_write], dim=-1)

@dataclass
class SplitGPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    d_resid_read: int = 768
    d_resid_write: int = 0
    dropout: float = 0.0
    per_layer_logit_coefficient: list[float] | None = None
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

    def __post_init__(self):
        if self.per_layer_logit_coefficient is None:
            self.per_layer_logit_coefficient = [0.0] * (self.n_layer) + [1.0]

    def get_normalized_coefficients_th(self) -> Float[torch.Tensor, "n_layer + 1"]:
        coeffs = torch.tensor(self.per_layer_logit_coefficient)
        return coeffs / coeffs.sum()

    @property 
    def d_resid(self):
        return self.d_resid_read + self.d_resid_write

    # Define alias for legacy reasons
    @property 
    def n_embd(self):
        return self.d_resid

class SplitGPTBlock(nn.Module):
    """ Basic transformer block """

    def __init__(self, config: SplitGPTConfig):
        super().__init__()
        self.config = config

        # Make sure attention reads only from the first d_resid dimensions
        self.c_ln1_pre = nn.Linear(config.d_resid_read, config.n_embd)
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config.n_head, config.n_embd, config.dropout, config.block_size, bias=config.bias)

        # Make sure MLP reads only from the first d_resid dimensions
        self.c_ln2_pre = nn.Linear(config.d_resid_read, config.n_embd)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config.n_embd, config.dropout)

    def forward(self, x: Float[torch.Tensor, "batch seq n_embd"]) -> Float[torch.Tensor, "batch seq n_embd"]:

        x_read, x_write = split_resid_into_read_and_write(x, self.config.d_resid_read, self.config.d_resid_write)

        # Attention
        ln1_in = self.c_ln1_pre(x_read)
        attn_in = self.ln_1(ln1_in)
        attn_out = self.attn(attn_in)
        attn_out_read, attn_out_write = split_resid_into_read_and_write(attn_out, self.config.d_resid_read, self.config.d_resid_write)
        x_read = x_read + attn_out_read
        x_write = x_write + attn_out_write

        # MLP
        ln2_in = self.c_ln2_pre(x_read)
        mlp_in = self.ln_2(ln2_in)
        mlp_out = self.mlp(mlp_in)
        mlp_out_read, mlp_out_write = split_resid_into_read_and_write(mlp_out, self.config.d_resid_read, self.config.d_resid_write)
        x_read = x_read + mlp_out_read
        x_write = x_write + mlp_out_write

        x = merge_resid_read_and_write(x_read, x_write)
        return x

# NOTE: Currently, there is a lot of code duplication between this and GPT
# TODO: Refactor the code to reduce duplication
# Undecided on what abstractions make sense here
class SplitGPT(nn.Module, Model):
    """ Same as GPT but with the following differences
    
    1. All blocks can only read from the first d_resid dimensions 
    2. The unembedding matrix can only read from the last d_resid_out dimensions
    3. The logits are aggregated from every layer instead of just the last layer
    """

    def __init__(self, config: SplitGPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.d_resid_read),
            wpe = nn.Embedding(config.block_size, config.d_resid_read),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([SplitGPTBlock(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.d_resid_write, bias=config.bias),
        ))
        # We need an initial value for the write stream, and it makes sense to use zeros
        self.register_buffer("init_resid_write", torch.zeros(config.block_size, config.d_resid_write))

        self.lm_head = nn.Linear(config.d_resid_write, config.vocab_size, bias=False)
        # GPT has weight tying baked in, so we should do that too 
        if not config.d_resid_write == config.d_resid_read:
            raise ValueError(f"d_resid_read {config.d_resid_read} != {config.d_resid_write}. Cannot tie weights.")
    
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, 
        idx: Int[torch.Tensor, "batch seq"], 
        targets: Int[torch.Tensor, "batch seq"] = None
    ) -> tuple[Float[torch.Tensor, "batch seq vocab_size"], Float[torch.Tensor, "batch"] | None]:        
        layerwise_x = self.get_layerwise_resid(idx)
        logits = self.get_logits(layerwise_x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            loss = None

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        raise NotImplementedError("from_pretrained not implemented for SplitGPT")
    
    def get_layerwise_resid(self, idx: Int[torch.Tensor, "batch seq"]) -> Float[torch.Tensor, "batch layer seq n_embd"]:
        """ Get the residual tensor at each layer """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, d_resid_read)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, d_resid_read)
        x_read = self.transformer.drop(tok_emb + pos_emb) # (b, t, d_resid_read)
        x_write = einops.repeat(self.init_resid_write, 't d -> b t d', b=b) # (b, t, d_resid_write)
        x = merge_resid_read_and_write(x_read, x_write) # (b, t, d_resid)

        layerwise_x = [x]
        for block in self.transformer.h:
            x = block(x) # each block has shape (b, t, n_embd)
            layerwise_x.append(x)
        
        return torch.stack(layerwise_x, dim=1) # (b, n_layer + 1, t, n_embd)

    def get_logits(
        self, 
        layerwise_x: Float[torch.Tensor, "batch layer seq n_embd"],
        coeffs: Float[torch.Tensor, "n_layer + 1"] | None = None
    ) -> Float[torch.Tensor, "batch seq vocab_size"]:
        """ Get the logits from the model 
        
        Aggregate logits are a weighted linear combination of the layer logits
        Coefficients are expected to sum to 1
        If None, the default coefficients from the config are used
        """
        if coeffs is None:
            coeffs = self.config.get_normalized_coefficients_th().to(layerwise_x.device)
        # Check coefficients
        assert coeffs.size(0) == layerwise_x.size(1), f"Expected {layerwise_x.size(1)} coefficients; got {coeffs.size(0)}"
        assert torch.allclose(coeffs.sum(), torch.tensor(1.0, device=layerwise_x.device)), f"Expected coefficients to sum to 1; got {coeffs.sum()}"
        
        _, layerwise_x_write = split_resid_into_read_and_write(layerwise_x, self.config.d_resid_read, self.config.d_resid_write)
        layerwise_x_normed = self.transformer.ln_f(layerwise_x_write) # (b, n_layer + 1, t, d_resid_write)        
        agg_x = einops.einsum(layerwise_x_normed, coeffs, 'b l t d, l -> b t d')    
        logits = self.lm_head(agg_x) # (b, t, vocab_size)
        return logits
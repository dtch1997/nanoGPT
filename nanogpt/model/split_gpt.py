import math
import inspect
import einops
import torch 
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from jaxtyping import Float, Int

from nanogpt.nn import LayerNorm, CausalSelfAttention, MLP

def split_resid_into_read_and_write(
    x: Float[torch.Tensor, "... d_resid"], 
    d_resid_read: int,
    d_resid_write: int
) -> tuple[Float[torch.Tensor, "... d_resid_read"], Float[torch.Tensor, "... d_resid_write"]]:
    """ Split the residual stream into the read and write streams """
    d_resid = x.shape[-1]
    assert d_resid_read + d_resid_write == d_resid, f"Expected d_resid_read + d_resid_write = d_resid; got {d_resid_read} + {d_resid_write} != {d_resid}"
    return x.tensor_split([d_resid_read, d_resid_write], dim=-1)

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
    per_layer_logit_coefficient: list[float] = [1.0] * 13 # (n_layer + 1) coefficients for each layer's logits
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

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
        self.attn = CausalSelfAttention(config.n_head, config.d_resid_read, config.dropout, config.block_size, bias=config.bias)
        self.c_attn_post = nn.Linear(config.n_embd, config.d_resid_read)

        # Make sure MLP reads only from the first d_resid dimensions
        self.c_ln2_pre = nn.Linear(config.d_resid_read, config.n_embd)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config.n_embd, config.dropout)
        self.c_mlp_post = nn.Linear(config.n_embd, config.d_resid_read)

    def forward(self, x: Float[torch.Tensor, "batch seq n_embd"]) -> Float[torch.Tensor, "batch seq n_embd"]:

        x_read, x_write = split_resid_into_read_and_write(x, self.config.d_resid_read, self.config.d_resid_write)

        # Attention
        ln1_in = self.c_ln1_pre(x_read)
        attn_in = self.ln_1(ln1_in)
        attn_out = self.attn(attn_in)
        x_read = x_read + self.c_attn_post(attn_out)

        # MLP
        ln2_in = self.c_ln2_pre(x_read)
        mlp_in = self.ln_2(ln2_in)
        mlp_out = self.mlp(mlp_in)
        x_read = x_read + self.c_mlp_post(mlp_out)

        return merge_resid_read_and_write(x_read, x_write)

# NOTE: Currently, there is a lot of code duplication between this and GPT
# TODO: Refactor the code to reduce duplication
# Undecided on what abstractions make sense here
class SplitGPT(nn.Module):
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

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

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
        agg_x = torch.einsum('b l t d, l -> b t d', layerwise_x_normed, coeffs) # (b, t, d_resid_write)
        logits = self.lm_head(agg_x) # (b, t, vocab_size)
        return logits

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

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = SplitGPTConfig(**config_args)
        model = SplitGPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
import pytest
import torch
import torch.nn as nn
import math

from nanogpt.nn import CausalSelfAttention, MLP

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TestCausalSelfAttention:
    @pytest.fixture
    def csa_params(self):
        return {
            "n_head": 4,
            "d_model": 64,
            "dropout": 0.1,
            "block_size": 16,
            "bias": True
        }
    
    def test_causal_self_attention(self, csa_params, device):
        csa = CausalSelfAttention(**csa_params).to(device)
        x = torch.randn(2, csa_params["block_size"], csa_params["d_model"]).to(device)
        
        output = csa(x)
        
        assert output.shape == (2, csa_params["block_size"], csa_params["d_model"])
    
    def test_causal_mask(self, csa_params, device):
        csa = CausalSelfAttention(**csa_params).to(device)
        x = torch.randn(1, csa_params["block_size"], csa_params["d_model"]).to(device)
        
        with torch.no_grad():
            q, k, v = csa.c_attn(x).split(csa.d_model, dim=2)
            q = q.view(1, csa_params["block_size"], csa.n_head, -1).transpose(1, 2)
            k = k.view(1, csa_params["block_size"], csa.n_head, -1).transpose(1, 2)
            
            attn_weights = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            mask = torch.triu(torch.ones_like(attn_weights), diagonal=1).bool()
            
            if not csa.flash:
                assert torch.all(attn_weights.masked_fill(csa.causal_attn_mask[:,:,:csa_params["block_size"],:csa_params["block_size"]] == 0, float('-inf')).masked_select(mask) == float('-inf'))

    def test_d_model_n_head_assertion(self):
        with pytest.raises(AssertionError):
            CausalSelfAttention(n_head=3, d_model=64, dropout=0.1, block_size=16)

class TestMLP:
    @pytest.fixture
    def mlp_params(self):
        return {
            "d_model": 64,
            "dropout": 0.1
        }
    
    def test_mlp(self, mlp_params, device):
        mlp = MLP(**mlp_params).to(device)
        x = torch.randn(5, 10, mlp_params["d_model"]).to(device)
        
        output = mlp(x)
        
        assert output.shape == (5, 10, mlp_params["d_model"])
    
    def test_mlp_activation(self, mlp_params, device):
        mlp = MLP(**mlp_params).to(device)
        x = torch.randn(1, 1, mlp_params["d_model"]).to(device)
        
        with torch.no_grad():
            fc1_output = mlp.c_fc(x)
            gelu_output = mlp.gelu(fc1_output)
            
            assert torch.all(gelu_output >= -1)  # GELU should always output a value greater than -1
    
    def test_mlp_dimensions(self, mlp_params, device):
        mlp = MLP(**mlp_params).to(device)
        
        assert mlp.c_fc.in_features == mlp_params["d_model"]
        assert mlp.c_fc.out_features == 4 * mlp_params["d_model"]
        assert mlp.c_proj.in_features == 4 * mlp_params["d_model"]
        assert mlp.c_proj.out_features == mlp_params["d_model"]

if __name__ == "__main__":
    pytest.main()
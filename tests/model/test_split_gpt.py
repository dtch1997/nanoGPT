import pytest
import torch
from nanogpt.model.split_gpt import SplitGPT, SplitGPTConfig

@pytest.fixture
def model_and_config() -> tuple[SplitGPT, SplitGPTConfig]:
    config = SplitGPTConfig(
        block_size=16,
        vocab_size=1000,
        n_layer=4,
        n_head=4,
        d_resid_read=128,
        d_resid_write=128,
        dropout=0.1,
    )
    model = SplitGPT(config)
    return model, config

@pytest.fixture
def sample_input(model_and_config:  tuple[SplitGPT, SplitGPTConfig]) -> torch.Tensor:
    _, config = model_and_config
    batch_size = 2
    seq_length = 16
    return torch.randint(0, config.vocab_size, (batch_size, seq_length))

def test_get_layerwise_logits_not_all_same(model_and_config:  tuple[SplitGPT, SplitGPTConfig], sample_input):
    model, _ = model_and_config
    layerwise_resid = model.get_layerwise_resid(sample_input)
    
    # Check that all resids are different
    for i in range(1, layerwise_resid.shape[1]):
        for j in range(i):
            assert not torch.allclose(layerwise_resid[:, i], layerwise_resid[:, j]), \
                f"Layerwise resids {i} and {j} are the same"

def test_all_parameters_receive_gradients(model_and_config:  tuple[SplitGPT, SplitGPTConfig], sample_input):
    model, _ = model_and_config
    targets = torch.randint(0, model.config.vocab_size, sample_input.shape)
    
    # Forward pass
    logits, loss = model(sample_input, targets)
    
    # Backward pass
    loss.backward()
    
    # Check that all parameters have gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
        assert torch.sum(param.grad ** 2) > 0, f"Parameter {name} has zero gradient"

def test_output_shape(model_and_config:  tuple[SplitGPT, SplitGPTConfig], sample_input):
    model, config = model_and_config
    logits, _ = model(sample_input)
    
    expected_shape = (sample_input.shape[0], sample_input.shape[1], config.vocab_size)
    assert logits.shape == expected_shape, f"Expected output shape {expected_shape}, but got {logits.shape}"

def test_loss_calculation(model_and_config:  tuple[SplitGPT, SplitGPTConfig], sample_input):
    model, _ = model_and_config
    targets = torch.randint(0, model.config.vocab_size, sample_input.shape)
    
    _, loss = model(sample_input, targets)
    assert isinstance(loss, torch.Tensor), "Loss should be a torch.Tensor"
    assert loss.ndim == 0, "Loss should be a scalar"
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert not torch.isinf(loss), "Loss should not be infinite"

def test_model_training_step(model_and_config:  tuple[SplitGPT, SplitGPTConfig], sample_input):
    model, _ = model_and_config
    targets = torch.randint(0, model.config.vocab_size, sample_input.shape)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Initial forward pass
    _, initial_loss = model(sample_input, targets)
    
    # Training step
    optimizer.zero_grad()
    _, loss = model(sample_input, targets)
    loss.backward()
    optimizer.step()
    
    # Check that loss decreased
    assert loss < initial_loss, "Loss should decrease after a training step"

if __name__ == "__main__":
    pytest.main()
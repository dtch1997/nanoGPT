import pytest
import torch

from nanogpt.model.split_gpt import SplitGPT, SplitGPTConfig, SplitGPTBlock
from nanogpt.model.split_gpt import split_resid_into_read_and_write, merge_resid_read_and_write

@pytest.fixture
def block_and_config():
    config = SplitGPTConfig(
        block_size=32,
        vocab_size=100,
        n_layer=1,
        n_head=2,
        d_resid_read=32,
        d_resid_write=32,
        dropout=0.0 # No dropout for testing
    )
    block = SplitGPTBlock(config)
    return block, config

def test_splitgpt_block_ignores_input_write_stream(block_and_config):
    block, config = block_and_config
    
    # Create two inputs with same read stream but different write streams
    x_read = torch.randn(2, 16, config.d_resid_read)
    x_write1 = torch.randn(2, 16, config.d_resid_write)
    x_write2 = torch.randn(2, 16, config.d_resid_write)

    x1 = torch.cat([x_read, x_write1], dim=-1)
    x2 = torch.cat([x_read, x_write2], dim=-1)
    
    # Process both inputs
    out1 = block(x1)
    out2 = block(x2)

    actual_read_diff = out1[:, :, :config.d_resid_read] - out2[:, :, :config.d_resid_read]
    expected_read_diff = torch.zeros_like(actual_read_diff)

    # Check that the read stream is the same
    assert torch.allclose(actual_read_diff, expected_read_diff, atol=1e-6), \
        "SplitGPTBlock should ignore the write stream"
    
    # Check that the write stream differs by the expected amount
    expected_write_diff = x2[:, :, config.d_resid_read:] - x1[:, :, config.d_resid_read:]
    actual_write_diff = out2[:, :, config.d_resid_read:] - out1[:, :, config.d_resid_read:]
    assert torch.allclose(actual_write_diff, expected_write_diff, atol=1e-6), \
        "SplitGPTBlock should modify the write stream by the expected amount"


def test_splitgpt_block_modifies_write_stream(block_and_config):
    block, config = block_and_config
    
    # Create input
    x = torch.randn(2, 16, config.d_resid)
    
    # Process input
    out = block(x)
    
    # Check that the write stream has been modified
    assert not torch.allclose(x[:, :, config.d_resid_read:], out[:, :, config.d_resid_read:], atol=1e-6), \
        "SplitGPTBlock should modify the write stream"
    
    # Check that the read stream has been modified (as it should be)
    assert not torch.allclose(x[:, :, :config.d_resid_read], out[:, :, :config.d_resid_read], atol=1e-6), \
        "SplitGPTBlock should modify the read stream"

def test_split_merge():
    # Set up test data
    d_resid = 128
    d_resid_read = 64
    d_resid_write = 64
    x = torch.randn(2, 16, d_resid, requires_grad=True)
    
    # Forward pass
    x_read, x_write = split_resid_into_read_and_write(x, d_resid_read, d_resid_write)
    x_merged = merge_resid_read_and_write(x_read, x_write)

    # Check that x_merged is the same as x
    assert torch.allclose(x, x_merged), "Merged tensor should be the same as the original tensor"

    # Compute loss and backward
    loss = x_merged.sum()
    loss.backward()
    
    # Check that gradients are preserved
    assert x.grad is not None, "Input tensor should have gradients"
    assert torch.allclose(x.grad, torch.ones_like(x)), "Gradients should be all ones"
    
    # Check that gradients flow through both read and write streams
    assert torch.all(x.grad[:, :, :d_resid_read] > 0), "Read stream should have positive gradients"
    assert torch.all(x.grad[:, :, d_resid_read:] > 0), "Write stream should have positive gradients"
    
    # Check that the split tensors are on the computation graph
    assert x_read.requires_grad, "x_read should require gradients"
    assert x_write.requires_grad, "x_write should require gradients"

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

def test_layerwise_resid(model_and_config:  tuple[SplitGPT, SplitGPTConfig], sample_input):
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
        if not param.requires_grad:
            continue
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

@pytest.fixture
def model_and_data():
    config = SplitGPTConfig(
        block_size=32,
        vocab_size=100,
        n_layer=2,
        n_head=2,
        d_resid_read=32,
        d_resid_write=32,
        dropout=0.1,
    )
    model = SplitGPT(config)
    
    # Create sample data
    batch_size = 4
    seq_length = 32
    x = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    y = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    
    return model, x, y


if __name__ == "__main__":
    pytest.main()
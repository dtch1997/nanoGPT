import pathlib
import nnsight
import pickle
import torch
import numpy as np
import os

from jaxtyping import Float
from nnsight import NNsight

from nanogpt.model import GPT
from nanogpt.utils.checkpoint import load_checkpoint
from nanogpt.utils.device import get_device

this_dir = pathlib.Path(__file__).parent.absolute() # .../notebooks
project_dir = this_dir.parent 
data_dir = project_dir / 'data' / 'shakespeare_char'

def load_metadata():
    with open(data_dir / 'meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    return meta

def load_test_data():
    data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    return torch.from_numpy(data).to(torch.int64)

if __name__ == "__main__":

    device = get_device()

    # Load the tokenization info
    metadata = load_metadata()
    stoi = metadata['stoi']
    itos = metadata['itos']

    # Get some test data
    test_data = load_test_data()
    tokens = test_data[:256]
    # print the test data
    print("Test data:")
    print("".join([itos[i.item()] for i in tokens]))
    tokens = tokens.unsqueeze(0)
    tokens = tokens.to(device)

    # Load the model
    checkpoint = 'gelu-2l-111'
    model = load_checkpoint(project_dir / 'checkpoints' / checkpoint)
    print(checkpoint)

    def get_logits(resid_pre_write: Float[torch.Tensor, "batch seq d_vocab"]) -> Float[torch.Tensor, "batch seq d_vocab"]:
        """ Get the logits from the model """
        resid_normed = model.transformer.ln_f(resid_pre_write)
        logits = model.lm_head(resid_normed)
        return logits

    def get_layerwise_logits(model: GPT, input) -> Float[torch.Tensor, "batch layer seq d_vocab"]:
        """ Get the logits after each layer """

        n_layers = len(model.transformer.h)
        all_layer_logits = []

        _, cache = model.run_with_cache(input)
        
        for k in range(n_layers):
            layer_k_resid_pre_write = cache[f'transformer.h.{k}.resid_pre_y']
            layer_k_logits = get_logits(layer_k_resid_pre_write)
            all_layer_logits.append(layer_k_logits)

        # also add last layer
        final_layer_resid_post_write = cache[f'transformer.h.{n_layers-1}.resid_post_y']
        final_layer_logits = get_logits(final_layer_resid_post_write)
        all_layer_logits.append(final_layer_logits)

        return torch.stack(all_layer_logits, dim=1)
    
    layerwise_logits = get_layerwise_logits(model, tokens)
    # Convert to log probs
    layerwise_logprobs = torch.log_softmax(layerwise_logits, dim=-1)

    # calculate KL divergence
    final_layer_logprobs = layerwise_logprobs[:, -1]

    kl_divs = []
    for i in range(layerwise_logprobs.shape[1]):
        curr_layer_logprobs = layerwise_logprobs[:, i]
        kl_div = torch.nn.functional.kl_div(curr_layer_logprobs, final_layer_logprobs, reduction='mean', log_target=True)
        print(f"KL divergence between layer {i} and final layer: {kl_div.item()}")
        kl_divs.append(kl_div.item())
    kl_divs = torch.tensor(kl_divs)

    # Save the KL divergences
    save_dir = pathlib.Path('kl_div')
    save_dir.mkdir(exist_ok=True)
    torch.save(kl_divs, save_dir / f'{checkpoint}.pt')
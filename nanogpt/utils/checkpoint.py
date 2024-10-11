import os 
import torch 

from nanogpt.utils.device import get_device
from nanogpt.model import GPT, GPTConfig


def load_checkpoint(out_dir) -> GPT:
    """ Load a saved checkpoint from a directory """
    device = get_device()
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict: dict[str, torch.Tensor] = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    return model
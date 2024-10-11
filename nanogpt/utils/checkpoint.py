import pathlib 
import torch 

from nanogpt.utils.device import get_device
from nanogpt.model import GPT, GPTConfig


def load_checkpoint(checkpoint_dir: str | pathlib.Path) -> GPT:
    """ Load a saved checkpoint from a directory """
    device = get_device()

    # Parse the checkpoint path
    if isinstance(checkpoint_dir, str):
        checkpoint_dir = pathlib.Path(checkpoint_dir).absolute()
    assert checkpoint_dir.is_dir(), f"Directory not found: {checkpoint_dir}"
    checkpoint_path = checkpoint_dir / 'ckpt.pt'

    # Load the model from the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    model = model.to(device)
    state_dict: dict[str, torch.Tensor] = checkpoint['model']

    # Preprocess the prefixes in the state_dict
    # NOTE(dtch1997): I don't fully understand this part, but it was in the original codebase
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    return model
#!/bin/bash

# When settings are implemented correctly, SplitGPT should be identical to GPT

# This should reproduce a 2l model
python train_split_gpt.py config/train_gelu-2l-split_shakespeare-char.py --per_layer_logit_coefficient="[0,0,1]" --wandb_run_name=gelu-2l-split --d_resid_read=512 --d_resid_write=0
python train_gpt.py config/train_gelu-2l_shakespeare-char.py --wandb_run_name=gelu-2l

# This should reproduce a 1l model
python train_split_gpt.py config/train_gelu-2l-split_shakespeare-char.py --per_layer_logit_coefficient="[0,1,0]" --wandb_run_name=gelu-1l-split --d_resid_read=512 --d_resid_write=0
python train_gpt.py config/train_gelu-2l_shakespeare-char.py --wandb_run_name=gelu-2l --n_layer=1

# This should reproduce a 0l model
python train_split_gpt.py config/train_gelu-2l-split_shakespeare-char.py --per_layer_logit_coefficient="[1,0,0]" --wandb_run_name=gelu-0l-split--d_resid_read=512 --d_resid_write=0
python train_gpt.py config/train_gelu-2l_shakespeare-char.py --wandb_run_name=gelu-2l --n_layer=0

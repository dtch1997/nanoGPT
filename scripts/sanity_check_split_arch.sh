#!/bin/bash

# Script to train "split architecture" GeLU models on Shakespeare-char dataset

python train.py config/train_gelu-2l-mod_shakespeare-char.py --per_layer_weight="[0,1,0,0,0]" --wandb_run_name=gelu-1l-mod_shakespeare-char
python train.py config/train_gelu-2l-mod_shakespeare-char.py --per_layer_weight="[0,0,1,0,0]" --wandb_run_name=gelu-2l-mod_shakespeare-char
python train.py config/train_gelu-2l-mod_shakespeare-char.py --per_layer_weight="[0,0,0,0,1]" --wandb_run_name=gelu-4l-mod_shakespeare-char

#!/bin/bash

# Script to sweep different combinatinos of coefficients for the "split architecture" GeLU models on Shakespeare-char dataset

python train.py config/train_gelu-2l-mod_shakespeare-char.py --out_dir=gelu-2l-001 --n_layer=2 --per_layer_weight="[0,0,1]" --wandb_project=sweep_coefficients --wandb_run_name=gelu-2l-001
python train.py config/train_gelu-2l-mod_shakespeare-char.py --out_dir=gelu-2l-010 --n_layer=2 --per_layer_weight="[0,1,0]" --wandb_project=sweep_coefficients --wandb_run_name=gelu-2l-010
python train.py config/train_gelu-2l-mod_shakespeare-char.py --out_dir=gelu-2l-100 --n_layer=2 --per_layer_weight="[1,0,0]" --wandb_project=sweep_coefficients --wandb_run_name=gelu-2l-100
python train.py config/train_gelu-2l-mod_shakespeare-char.py --out_dir=gelu-2l-111 --n_layer=2 --per_layer_weight="[1,1,1]" --wandb_project=sweep_coefficients --wandb_run_name=gelu-2l-111

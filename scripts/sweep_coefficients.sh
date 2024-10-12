#!/bin/bash

# Script to sweep different combinatinos of coefficients for the "split architecture" GeLU models on Shakespeare-char dataset

python train_split_gpt.py config/train_gelu-2l-split_shakespeare-char.py --out_dir=gelu-2l-111 --n_layer=2 --per_layer_logit_coefficient="[1,1,1]" --wandb_project=sweep_coefficients --wandb_run_name=gelu-2l-111

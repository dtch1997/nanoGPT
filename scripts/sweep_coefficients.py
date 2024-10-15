""" Script to sweep different coefficients """

import shlex 
import os
import argparse

CONFIG_FILE = "config/train_gelu-2l.py"
WANDB_PROJECT = "interpretable-lms-3"

coefficients = {
    "001": "[0,0,1]", # simulates a 2l model
    "010": "[0,1,0]", # simulates a 1l model
    "100": "[1,0,0]", # simulates a 0l model
    "111": "[1,1,1]", # a mixed model
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_project", type=str, default=WANDB_PROJECT)
    parser.add_argument("--config_file", type=str, default=CONFIG_FILE)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    CONFIG_FILE = args.config_file
    WANDB_PROJECT = args.wandb_project

    for name, coeffs in coefficients.items():
        command_parts = [
            "python",
            "train.py",
            CONFIG_FILE,
            f"--n_layer=2",
            f"--out_dir=checkpoints/gelu-2l-{name}",
            f"--per_layer_weight={coeffs}",
            f"--wandb_run_name=gelu-2l-{name}",
            f"--wandb_project={WANDB_PROJECT}",
        ]
        command = " ".join(command_parts)
        print(command)
        if args.run: os.system(command)
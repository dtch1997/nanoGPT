""" Train a GELU-2L model on the TinyStories dataset 

Intended to produce artifacts for mech interp analysis """

# Metadata
out_dir = 'out-gelu-2l'
wandb_log = True
wandb_project = 'interpretable-lms'
wandb_run_name='gelu-2l_tinystories'

# Dataset details
dataset = 'tinystories'
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 1

# Model architecture
n_layer = 2
n_head = 8
n_embd = 512
dropout = 0.2
per_layer_weight = [0, 0, 1]

# Training stuff
max_iters = 20_000 # TODO: figure out how many iterations to run
lr_decay_iters = 20_000
weight_decay = 1e-1
always_save_checkpoint = False # we expect to overfit on this small dataset, so only save when val improves

# Eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# Miscellanous details
compile = False
""" Train a GELU-2L model on the TinyStories dataset """

# Metadata
out_dir = 'out-gelu-2l'
wandb_log = True
wandb_project = 'interpretable-lms'
wandb_run_name='gelu-2l_tinystories'

# Training details
dataset = 'tinystories'
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 1

# Model architecture
n_layer = 2
n_head = 8
n_embd = 512
dropout = 0.2

# Training stuff
max_iters = 600_000 # TODO: figure out how many iterations to run
lr_decay_iters = 600_000

# Eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1
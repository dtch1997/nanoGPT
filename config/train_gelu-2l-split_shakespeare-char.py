""" Train a GELU-2L model on the Shakespeare character dataset 

Intended mainly to sanity-check the implementation """

# Metadata
out_dir = 'out-gelu-2l-split-shakespeare-char'
wandb_log = True # override via command line if you like
wandb_project = 'interpretable-lms'
wandb_run_name = 'gelu-2l-split_shakespeare-char'

# Dataset details
dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

# Model architecture
n_layer = 2
n_head = 8
d_resid_read = 256
d_resid_write = 256
dropout = 0.2
per_layer_logit_coefficient = [0, 0, 1]

# Training details
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
learning_rate = 1e-3 # with baby networks can afford to go a bit higher
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small
warmup_iters = 100 # not super necessary potentially
always_save_checkpoint = False # we expect to overfit on this small dataset, so only save when val improves

# Eval details
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# Miscellanous details
compile = False
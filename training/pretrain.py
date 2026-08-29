# Main pytorch libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import Dataset, DataLoader
import os
import yaml
from model.transformer import Model
from training.trainer import train_model, load_checkpoint

# repo root, overridable so the same code runs on Colab
ROOT = os.environ.get('PROJECT_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# on Colab, copy the .bin files to local disk and set DATA_DIR=/content/data --
# memmapping them over a mounted Drive makes random reads unusably slow
DATA_DIR = os.environ.get('DATA_DIR', os.path.join(ROOT, 'data/processed'))

TRAIN_PATH = os.path.join(DATA_DIR, 'fineweb_train.bin')
VAL_PATH = os.path.join(DATA_DIR, 'fineweb_val.bin')

PRETRAIN_CONFIG_PATH = os.path.join(ROOT, 'configs/pretrain.yaml')
MODEL_CONFIG_PATH = os.path.join(ROOT, 'configs/model.yaml')
TOKENIZER_CONFIG_PATH = os.path.join(ROOT, 'configs/tokenizer.yaml')


# Get configs

with open(PRETRAIN_CONFIG_PATH, 'r') as f:
    PRETRAIN_CONFIG = yaml.safe_load(f)

with open(MODEL_CONFIG_PATH, 'r') as f:
    MODEL_CONFIG = yaml.safe_load(f)

with open(TOKENIZER_CONFIG_PATH, 'r') as f:
    TOKENIZER_CONFIG = yaml.safe_load(f)



CONTEXT_SIZE = MODEL_CONFIG['context_size']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
CHECKPOINT_DIR = os.path.join(ROOT, PRETRAIN_CONFIG['checkpoint_path'])


class TokenDataset(Dataset):
    def __init__(self, path, context_size):
        self.path = path
        self.context_size = context_size
        self.data = None
        self.length = os.path.getsize(path) // 2 // context_size - 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.data is None:
            self.data = np.memmap(self.path, dtype=np.uint16, mode='r')
        start = idx * self.context_size
        chunk = self.data[start:start + self.context_size + 1]
        return (torch.from_numpy(chunk[:-1].astype(np.int64)),
                torch.from_numpy(chunk[1:].astype(np.int64)))


def main():
    on_gpu = DEVICE.type == 'cuda'

    if on_gpu:
        # let the fp32 ops that remain (and anything outside autocast) use TF32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    train_dataset = TokenDataset(TRAIN_PATH, CONTEXT_SIZE)
    val_dataset = TokenDataset(VAL_PATH, CONTEXT_SIZE)

    num_workers = 6 if DEVICE.type != 'cpu' else 0
    loader_kwargs = dict(num_workers=num_workers, pin_memory=on_gpu, drop_last=True)
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = True  # don't respawn workers on every eval

    train_loader = DataLoader(train_dataset, batch_size=PRETRAIN_CONFIG['batch_size'],
                              shuffle=True, **loader_kwargs)

    val_loader = DataLoader(val_dataset, batch_size=PRETRAIN_CONFIG['batch_size'],
                            shuffle=False, **loader_kwargs)


    model = Model(d_embed=MODEL_CONFIG['d_embed'], vocab_size=TOKENIZER_CONFIG['vocab_size'],
                  max_ctx=MODEL_CONFIG['context_size'], num_layers=MODEL_CONFIG['num_layers'],
                  numHeads=MODEL_CONFIG['num_heads'], dropout=MODEL_CONFIG['dropout']).to(DEVICE)

    if on_gpu:
        # worth a minute of warm-up on an A100; skipped elsewhere since the
        # inductor backend isn't a reliable win on MPS/CPU
        model = torch.compile(model)

    # only decay matrices -- not norms, biases or embeddings
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = optim.AdamW(
        [{'params': decay, 'weight_decay': PRETRAIN_CONFIG['weight_decay']},
         {'params': no_decay, 'weight_decay': 0.0}],
        lr=PRETRAIN_CONFIG['learning_rate'], fused=on_gpu)

    linear_warmup = LinearLR(optimizer, start_factor=PRETRAIN_CONFIG['init_linear_factor'], end_factor=1, total_iters=PRETRAIN_CONFIG['num_warmup_steps'])
    cosine_cheduler = CosineAnnealingLR(optimizer=optimizer, T_max=PRETRAIN_CONFIG['MAX_TRAINING_STEPS'] - PRETRAIN_CONFIG['num_warmup_steps'], eta_min=PRETRAIN_CONFIG['learning_rate'] * 0.05)

    scheduler = SequentialLR(optimizer, schedulers=[linear_warmup, cosine_cheduler], milestones=[PRETRAIN_CONFIG['num_warmup_steps']])

    # pick up where a disconnected Colab session left off
    start_step = 0
    resume_path = os.path.join(CHECKPOINT_DIR, 'last.pt')
    if os.path.exists(resume_path):
        start_step = load_checkpoint(resume_path, model, optimizer, scheduler, DEVICE)
        print(f"resuming from {resume_path} at step {start_step:,}")
    
    n_params = sum(p.numel() for p in model.parameters())
    tokens = PRETRAIN_CONFIG['MAX_TRAINING_STEPS'] * PRETRAIN_CONFIG['batch_size'] * PRETRAIN_CONFIG['grad_accum_steps'] * CONTEXT_SIZE
    print(f"device {DEVICE} | {n_params/1e6:.1f}M params | {tokens/1e9:.2f}B token budget", flush=True)

    train_model(model=model, max_train_steps=PRETRAIN_CONFIG['MAX_TRAINING_STEPS'],train_loader=train_loader,
                val_loader=val_loader,optimizer=optimizer, scheduler=scheduler,
                grad_clip_mag=PRETRAIN_CONFIG['grad_clip_norm'],
                device=DEVICE, eval_iters=PRETRAIN_CONFIG['eval_iters'],
                eval_interval=PRETRAIN_CONFIG['eval_interval'],
                checkpoint_interval=PRETRAIN_CONFIG['checkpoint_interval'],
                checkpoint_path=CHECKPOINT_DIR,
                grad_accum_steps=PRETRAIN_CONFIG['grad_accum_steps'],
                start_step=start_step)


if __name__ == '__main__':
    main()

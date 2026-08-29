import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import Dataset, DataLoader
import os
import yaml
from model.transformer import Model
from training.trainer import train_model, load_checkpoint, load_pretrained_weights


# repo root, overridable so the same code runs on Colab
ROOT = os.environ.get('PROJECT_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.environ.get('DATA_DIR', os.path.join(ROOT, 'data/processed'))

TRAIN_PATH = os.path.join(DATA_DIR, 'shakespeare_train.bin')
VAL_PATH = os.path.join(DATA_DIR, 'shakespeare_val.bin')

FINETUNE_CONFIG_PATH = os.path.join(ROOT, 'configs/finetune.yaml')
MODEL_CONFIG_PATH = os.path.join(ROOT, 'configs/model.yaml')
TOKENIZER_CONFIG_PATH = os.path.join(ROOT, 'configs/tokenizer.yaml')

with open(FINETUNE_CONFIG_PATH, 'r') as f:
    FINETUNE_CONFIG = yaml.safe_load(f)

with open(MODEL_CONFIG_PATH, 'r') as f:
    MODEL_CONFIG = yaml.safe_load(f)

with open(TOKENIZER_CONFIG_PATH, 'r') as f:
    TOKENIZER_CONFIG = yaml.safe_load(f)


CONTEXT_SIZE = MODEL_CONFIG['context_size']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
# finetuning writes to its own directory so it never overwrites the pretrain run
CHECKPOINT_DIR = os.path.join(ROOT, FINETUNE_CONFIG['checkpoint_path'])
PRETRAINED_PATH = os.path.join(ROOT, FINETUNE_CONFIG['pretrained_checkpoint'])


# We can unpack whole .bin file since dataset isnt very big. Only around 470k tokens in training and 54k in validation
class TokenDataset(Dataset):
    def __init__(self, path, context_size):
        self.path = path
        self.context_size = context_size
        # dtype is required -- the .bin files are raw uint16, and without it
        # numpy reads them as float64 and gets both the values and count wrong
        self.data = np.fromfile(self.path, dtype=np.uint16)
        # number of non-overlapping windows, not number of tokens
        self.length = len(self.data) // context_size - 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        start = idx * self.context_size
        chunk = self.data[start:start + self.context_size + 1]
        inputs = torch.from_numpy(chunk[:-1].astype(np.int64))
        targets = torch.from_numpy(chunk[1:].astype(np.int64))
        return (inputs, targets)


def main():
    on_gpu = DEVICE.type == 'cuda'

    if on_gpu:
        # let the fp32 ops that remain (and anything outside autocast) use TF32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')

    # make checkpoint dir if it doesnt exist already
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    train_dataset = TokenDataset(TRAIN_PATH, CONTEXT_SIZE)
    val_dataset = TokenDataset(VAL_PATH, CONTEXT_SIZE)

    # both splits are already in RAM, so worker processes would only add overhead
    loader_kwargs = dict(num_workers=0, pin_memory=on_gpu, drop_last=True)

    # init train and validation loaders
    train_loader = DataLoader(train_dataset, batch_size=FINETUNE_CONFIG['batch_size'],
                              shuffle=True, **loader_kwargs)

    val_loader = DataLoader(val_dataset, batch_size=FINETUNE_CONFIG['batch_size'],
                            shuffle=False, **loader_kwargs)

    # Model has to be built exactly as it was pretrained so the weights line up
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

    # Initialize optimizer, also ensuring weight decay only applies to matricies
    optimizer = optim.AdamW(
        [{'params': decay, 'weight_decay': FINETUNE_CONFIG['weight_decay']},
         {'params': no_decay, 'weight_decay': 0.0}],
        lr=FINETUNE_CONFIG['learning_rate'], fused=on_gpu)


    linear_warmup = LinearLR(optimizer, start_factor=FINETUNE_CONFIG['init_linear_factor'], end_factor=1, total_iters=FINETUNE_CONFIG['num_warmup_steps'])
    cosine_cheduler = CosineAnnealingLR(optimizer=optimizer, T_max=FINETUNE_CONFIG['MAX_TRAINING_STEPS'] - FINETUNE_CONFIG['num_warmup_steps'], eta_min=FINETUNE_CONFIG['learning_rate'] * 0.05)

    scheduler = SequentialLR(optimizer, schedulers=[linear_warmup, cosine_cheduler], milestones=[FINETUNE_CONFIG['num_warmup_steps']])

    # An interrupted finetune resumes itself; otherwise we start from the
    # pretrained weights. Without this the run would train from scratch.
    start_step = 0
    resume_path = os.path.join(CHECKPOINT_DIR, 'last.pt')
    if os.path.exists(resume_path):
        start_step = load_checkpoint(resume_path, model, optimizer, scheduler, DEVICE)
        print(f"resuming finetune from {resume_path} at step {start_step:,}")
    else:
        if not os.path.exists(PRETRAINED_PATH):
            raise FileNotFoundError(
                f"no pretrained checkpoint at {PRETRAINED_PATH} -- run training/pretrain.py first")
        load_pretrained_weights(PRETRAINED_PATH, model, DEVICE)
        print(f"loaded pretrained weights from {PRETRAINED_PATH}")

    n_params = sum(p.numel() for p in model.parameters())
    tokens = FINETUNE_CONFIG['MAX_TRAINING_STEPS'] * FINETUNE_CONFIG['batch_size'] * FINETUNE_CONFIG['grad_accum_steps'] * CONTEXT_SIZE
    epochs = tokens / (len(train_dataset) * CONTEXT_SIZE)
    print(f"device {DEVICE} | {n_params/1e6:.1f}M params | {tokens/1e6:.1f}M tokens (~{epochs:.1f} epochs)", flush=True)

    train_model(model=model, max_train_steps=FINETUNE_CONFIG['MAX_TRAINING_STEPS'], train_loader=train_loader,
                val_loader=val_loader, optimizer=optimizer, scheduler=scheduler,
                grad_clip_mag=FINETUNE_CONFIG['grad_clip_norm'],
                device=DEVICE, eval_iters=FINETUNE_CONFIG['eval_iters'],
                eval_interval=FINETUNE_CONFIG['eval_interval'],
                checkpoint_interval=FINETUNE_CONFIG['checkpoint_interval'],
                checkpoint_path=CHECKPOINT_DIR,
                grad_accum_steps=FINETUNE_CONFIG['grad_accum_steps'],
                start_step=start_step)


if __name__ == '__main__':
    main()

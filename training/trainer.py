import os
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_


def amp_context(device):
    #bf16 autocast on CUDA (A100), plain fp32 everywhere else.
    if device.type == 'cuda':
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    return nullcontext()


def unwrap(model):
    #torch.compile wraps the module; checkpoint the original so the saved keys
    #have no '_orig_mod.' prefix and stay loadable without compiling.
    return getattr(model, '_orig_mod', model)


def save_checkpoint(path, model, optimizer, scheduler, step):
    #Full training state, written atomically so a disconnect can't corrupt it.
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    torch.save({
        'model': unwrap(model).state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'step': step,
    }, tmp)
    os.replace(tmp, path)


def load_pretrained_weights(path, model, device):
    #Load only the model weights from a checkpoint. Used to start finetuning:
    #the optimizer and scheduler state belong to the pretraining run and must
    #not carry over, since finetuning has its own LR schedule.
    ckpt = torch.load(path, map_location=device, weights_only=False)
    unwrap(model).load_state_dict(ckpt['model'])


def load_checkpoint(path, model, optimizer, scheduler, device):
    #Restore a run saved by save_checkpoint. Returns the step it stopped at.
    ckpt = torch.load(path, map_location=device, weights_only=False)
    unwrap(model).load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    return ckpt['step']


def estimate_loss(model, val_loader, eval_iters, device, val_iter=None):
    model.eval()
    losses = []
    if val_iter is None:
        val_iter = iter(val_loader)
    amp = amp_context(device)
    with torch.no_grad():
        for _ in range(eval_iters):
            try:
                xb, yb = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                xb, yb = next(val_iter)
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with amp:
                logits = model(xb)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses), val_iter

def train_model(model, max_train_steps, train_loader, val_loader, optimizer, scheduler,
                grad_clip_mag, device, eval_interval, eval_iters,
                checkpoint_interval, checkpoint_path, grad_accum_steps, start_step=0):
    model.train()
    window_loss = 0.0
    window_steps = 0
    train_iter = iter(train_loader)
    val_iter = None
    amp = amp_context(device)

    for step in range(start_step + 1, max_train_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        accum_loss = torch.zeros((), device=device)

        for _ in range(grad_accum_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with amp:
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            # scale so accumulated grads average rather than sum
            (loss / grad_accum_steps).backward()
            # keep this on-device: calling .item() here would sync every micro-batch
            accum_loss += loss.detach() / grad_accum_steps

        grad_norm = clip_grad_norm_(model.parameters(), max_norm=grad_clip_mag)
        optimizer.step()
        scheduler.step()

        window_loss += accum_loss.item()
        window_steps += 1

        if step % eval_interval == 0:
            val_loss, val_iter = estimate_loss(model, val_loader, eval_iters, device, val_iter)
            print(f"step {step:,}/{max_train_steps:,} | train {window_loss/window_steps:.4f} | val {val_loss:.4f} | lr {scheduler.get_last_lr()[0]:.2e} | gnorm {grad_norm:.2f}", flush=True)
            window_loss = 0.0
            window_steps = 0

        if step % checkpoint_interval == 0:
            # single rolling checkpoint: full state is ~3x the model size, so
            # keeping one per interval would fill a Drive quota fast
            save_checkpoint(os.path.join(checkpoint_path, "last.pt"), model, optimizer, scheduler, step)

    save_checkpoint(os.path.join(checkpoint_path, "final.pt"), model, optimizer, scheduler, max_train_steps)

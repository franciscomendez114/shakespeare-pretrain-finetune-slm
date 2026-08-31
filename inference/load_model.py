import json
import os

import torch
import yaml

from model.transformer import Model
from tokenizer.bpe import BPE_Tokenizer


# repo root, overridable so the same code runs on Colab
ROOT = os.environ.get('PROJECT_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_CONFIG_PATH = os.path.join(ROOT, 'configs/model.yaml')
TOKENIZER_CONFIG_PATH = os.path.join(ROOT, 'configs/tokenizer.yaml')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def load_tokenizer():
    with open(TOKENIZER_CONFIG_PATH, 'r') as f:
        tokenizer_config = yaml.safe_load(f)
    return BPE_Tokenizer.from_pretrained(os.path.join(ROOT, tokenizer_config['artifacts_dir']))


def load_model(checkpoint_path, device=DEVICE):
    # Build the model exactly as it was trained, then load the saved weights.
    # Checkpoints hold {model, optimizer, scheduler, step} -- only the weights
    # are needed here.
    with open(MODEL_CONFIG_PATH, 'r') as f:
        model_config = yaml.safe_load(f)
    with open(TOKENIZER_CONFIG_PATH, 'r') as f:
        tokenizer_config = yaml.safe_load(f)

    model = Model(d_embed=model_config['d_embed'],
                  vocab_size=tokenizer_config['vocab_size'],
                  max_ctx=model_config['context_size'],
                  num_layers=model_config['num_layers'],
                  numHeads=model_config['num_heads'],
                  dropout=model_config['dropout']).to(device)

    checkpoint = torch.load(os.path.join(ROOT, checkpoint_path), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])

    # eval() turns off dropout -- without it every sample would be noisy
    model.eval()
    return model


def load_exported_model(export_dir, device=DEVICE):
    # Load a directory produced by scripts/export_model.py. Unlike load_model()
    # this reads nothing from configs/ -- everything it needs is in config.json,
    # so the export can be deployed on its own.
    export_dir = os.path.join(ROOT, export_dir) if not os.path.isabs(export_dir) else export_dir

    with open(os.path.join(export_dir, 'config.json'), 'r') as f:
        config = json.load(f)

    model = Model(d_embed=config['d_embed'],
                  vocab_size=config['vocab_size'],
                  max_ctx=config['context_size'],
                  num_layers=config['num_layers'],
                  numHeads=config['num_heads'],
                  dropout=0.0)

    weights = torch.load(os.path.join(export_dir, 'model.pt'), map_location='cpu', weights_only=True)

    # fp16 is stored to keep the file small, but CPU kernels for it are slow or
    # missing, so cast back to fp32 unless we are on a GPU
    if device.type == 'cpu':
        weights = {k: v.float() for k, v in weights.items()}

    model.load_state_dict(weights)
    model.to(device).eval()
    return model


def load_exported_tokenizer(export_dir):
    export_dir = os.path.join(ROOT, export_dir) if not os.path.isabs(export_dir) else export_dir
    return BPE_Tokenizer.from_pretrained(os.path.join(export_dir, 'tokenizer'))

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

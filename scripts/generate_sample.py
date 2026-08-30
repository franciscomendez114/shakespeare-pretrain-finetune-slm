import argparse
import sys
from pathlib import Path

# make the repo importable when this file is run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from inference.generate import generate
from inference.load_model import DEVICE, load_model, load_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/finetune/final.pt",
                        help="path relative to the repo root")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    tokenizer = load_tokenizer()
    model = load_model(args.checkpoint, DEVICE)
    print(f"{args.checkpoint} on {DEVICE} | temp {args.temperature} | top-k {args.top_k}\n")

    for i in range(args.samples):
        if args.samples > 1:
            print(f"----- sample {i + 1} -----")
        print(generate(model, tokenizer, prompt=args.prompt, max_new_tokens=args.tokens,
                       temperature=args.temperature, top_k=args.top_k, device=DEVICE))
        print()


if __name__ == "__main__":
    main()

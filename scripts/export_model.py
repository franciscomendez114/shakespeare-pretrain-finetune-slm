import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(
        description="Strip a training checkpoint down to what inference needs.")
    parser.add_argument("--checkpoint", default="checkpoints/finetune/best.pt",
                        help="path relative to the repo root")
    parser.add_argument("--out", default="export", help="output directory")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"],
                        help="fp16 halves the file and is fine for inference")
    args = parser.parse_args()

    src = ROOT / args.checkpoint
    if not src.exists():
        raise FileNotFoundError(f"no checkpoint at {src}")

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = yaml.safe_load((ROOT / "configs/model.yaml").read_text())
    tok_cfg = yaml.safe_load((ROOT / "configs/tokenizer.yaml").read_text())

    # a training checkpoint is {model, optimizer, scheduler, step, best_val}.
    # Only the weights matter here -- the optimizer moments are 2/3 of the file.
    checkpoint = torch.load(src, map_location="cpu", weights_only=False)
    weights = checkpoint["model"]

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    weights = {k: v.to(dtype) for k, v in weights.items()}

    # the head and the token embedding are the same matrix. Casting above made
    # two copies of it, so point them at one tensor again -- torch.save then
    # stores the shared storage once instead of twice.
    if "head.weight" in weights and torch.equal(weights["head.weight"], weights["token_embedding.weight"]):
        weights["head.weight"] = weights["token_embedding.weight"]

    torch.save(weights, out_dir / "model.pt")

    # everything needed to rebuild the model, so deployment does not read configs/
    config = {
        "d_embed": model_cfg["d_embed"],
        "num_layers": model_cfg["num_layers"],
        "num_heads": model_cfg["num_heads"],
        "context_size": model_cfg["context_size"],
        "vocab_size": tok_cfg["vocab_size"],
        "dtype": args.dtype,
        "source_checkpoint": args.checkpoint,
        "trained_steps": checkpoint.get("step"),
        "best_val_loss": checkpoint.get("best_val"),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    # the tokenizer travels with the weights; a model is useless without it
    tok_src = ROOT / tok_cfg["artifacts_dir"]
    tok_dst = out_dir / "tokenizer"
    if tok_dst.exists():
        shutil.rmtree(tok_dst)
    shutil.copytree(tok_src, tok_dst)

    before = src.stat().st_size
    after = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
    print(f"checkpoint {before/1e6:,.0f} MB  ->  export {after/1e6:,.0f} MB  ({before/after:.1f}x smaller)")
    print(f"  {out_dir}/model.pt        {(out_dir/'model.pt').stat().st_size/1e6:,.0f} MB ({args.dtype})")
    print(f"  {out_dir}/config.json")
    print(f"  {out_dir}/tokenizer/      {len(list(tok_dst.iterdir()))} files")
    if config["best_val_loss"] is not None:
        print(f"  step {config['trained_steps']}, best val {config['best_val_loss']:.4f}")


if __name__ == "__main__":
    main()

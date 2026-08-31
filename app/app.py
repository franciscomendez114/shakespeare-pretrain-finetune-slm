import os
import sys
from pathlib import Path

# so the app can import model/, inference/ and tokenizer/ from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr
import torch

from inference.generate import generate_stream
from inference.load_model import load_exported_model, load_exported_tokenizer


# Point MODEL_DIR at a local folder to run without downloading; otherwise the
# weights come from the Hugging Face repo named by HF_REPO.
MODEL_DIR = os.environ.get("MODEL_DIR")
HF_REPO = os.environ.get("HF_REPO", "franciscomendez114/shakespeare-slm")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FINETUNED = "Fine-tuned on Shakespeare"
PRETRAINED = "Pretrained on FineWeb-Edu"
SUBDIR = {FINETUNED: "finetuned", PRETRAINED: "pretrained"}

_loaded = {}


def model_root():
    if MODEL_DIR:
        return Path(MODEL_DIR)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=HF_REPO, repo_type="model"))


def get_model(choice):
    # loaded on first use and kept, so switching back and forth is instant
    if choice not in _loaded:
        path = model_root() / SUBDIR[choice]
        _loaded[choice] = (load_exported_model(str(path), DEVICE),
                           load_exported_tokenizer(str(path)))
    return _loaded[choice]


def run(choice, prompt, max_new_tokens, temperature, top_k):
    model, tokenizer = get_model(choice)
    for text in generate_stream(model, tokenizer, prompt=prompt,
                                max_new_tokens=int(max_new_tokens),
                                temperature=float(temperature),
                                top_k=int(top_k), device=DEVICE):
        yield text


with gr.Blocks(title="Shakespeare SLM") as demo:
    gr.Markdown(
        "# Shakespeare SLM\n"
        "A 172.6M-parameter transformer written from scratch: custom byte-level BPE "
        "tokenizer, pretrained on 3.45B tokens of FineWeb-Edu, then fine-tuned on "
        "tiny Shakespeare.\n\n"
        "Switch between the two checkpoints to see what fine-tuning changed. The "
        "pretrained model treats `ROMEO:` as ordinary web text; the fine-tuned one "
        "answers in play format."
    )

    with gr.Row():
        with gr.Column(scale=2):
            choice = gr.Radio(list(SUBDIR), value=FINETUNED, label="Model")
            prompt = gr.Textbox(label="Prompt", value="ROMEO:", lines=3)
            go = gr.Button("Generate", variant="primary")
        with gr.Column(scale=1):
            max_new_tokens = gr.Slider(20, 500, value=200, step=10, label="Tokens to generate")
            temperature = gr.Slider(0.1, 1.5, value=0.9, step=0.05, label="Temperature",
                                    info="Below ~0.8 this model repeats itself")
            top_k = gr.Slider(1, 500, value=100, step=1, label="Top-k")

    out = gr.Textbox(label="Output", lines=18)

    gr.Examples(
        [[FINETUNED, "ROMEO:", 200, 0.9, 100],
         [FINETUNED, "JULIET:\nO gentle", 200, 0.9, 100],
         [FINETUNED, "First Citizen:", 200, 0.85, 100],
         [PRETRAINED, "The mitochondria", 200, 0.9, 100],
         [PRETRAINED, "ROMEO:", 200, 0.9, 100]],
        inputs=[choice, prompt, max_new_tokens, temperature, top_k],
    )

    go.click(run, [choice, prompt, max_new_tokens, temperature, top_k], out)

if __name__ == "__main__":
    demo.launch()

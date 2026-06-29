"""
Plot training curves from the log history saved by train.py.

Reads <output_dir>/log_history.json and writes:
  <output_dir>/training_curves.png   -- loss, perplexity, and LR vs step

Usage
-----
    python plot_training.py --log_history output/log_history.json --output_dir output
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")  # headless / no display
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--log_history", required=True, help="log_history.json from train.py")
    p.add_argument("--output_dir",  required=True)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.log_history) as f:
        history = json.load(f)

    # split into train logs (have 'loss') and eval logs (have 'eval_loss')
    tr_steps,  tr_loss,  tr_ppl  = [], [], []
    ev_steps,  ev_loss,  ev_ppl  = [], [], []
    lr_steps,  lr_vals           = [], []

    for e in history:
        step = e.get("step")
        if "loss" in e:
            tr_steps.append(step); tr_loss.append(e["loss"])
            if "train_perplexity" in e:
                tr_ppl.append(e["train_perplexity"])
            else:
                tr_ppl.append(None)
        if "learning_rate" in e:
            lr_steps.append(step); lr_vals.append(e["learning_rate"])
        if "eval_loss" in e:
            ev_steps.append(step); ev_loss.append(e["eval_loss"])
            ev_ppl.append(e.get("eval_perplexity"))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── 1. Loss ───────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(tr_steps, tr_loss, "-o", ms=3, label="train loss", color="tab:blue")
    if ev_steps:
        ax.plot(ev_steps, ev_loss, "-s", ms=4, label="eval loss", color="tab:red")
    ax.set_xlabel("step"); ax.set_ylabel("cross-entropy loss")
    ax.set_title("Training & Eval Loss"); ax.grid(True, alpha=0.3); ax.legend()

    # ── 2. Perplexity ───────────────────────────────────────────────────────────
    ax = axes[1]
    tp = [(s, p) for s, p in zip(tr_steps, tr_ppl) if p is not None]
    if tp:
        ax.plot([s for s, _ in tp], [p for _, p in tp], "-o", ms=3,
                label="train ppl", color="tab:blue")
    ep = [(s, p) for s, p in zip(ev_steps, ev_ppl) if p is not None]
    if ep:
        ax.plot([s for s, _ in ep], [p for _, p in ep], "-s", ms=4,
                label="eval ppl", color="tab:red")
    ax.set_xlabel("step"); ax.set_ylabel("perplexity")
    ax.set_title("Perplexity"); ax.grid(True, alpha=0.3); ax.legend()

    # ── 3. Learning rate ─────────────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(lr_steps, lr_vals, "-", color="tab:green")
    ax.set_xlabel("step"); ax.set_ylabel("learning rate")
    ax.set_title("LR Schedule (cosine)"); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_png = os.path.join(args.output_dir, "training_curves.png")
    fig.savefig(out_png, dpi=130)
    print(f"Saved plot -> {out_png}")

    # quick text summary
    if tr_loss:
        print(f"train loss: {tr_loss[0]:.3f} (step {tr_steps[0]}) "
              f"-> {tr_loss[-1]:.3f} (step {tr_steps[-1]})")
    if ev_loss:
        print(f"eval  loss: {ev_loss[0]:.3f} -> {ev_loss[-1]:.3f}")


if __name__ == "__main__":
    main()

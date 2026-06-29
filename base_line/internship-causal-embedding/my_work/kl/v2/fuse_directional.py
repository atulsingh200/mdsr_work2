"""Fuse the trained CDEv2-directional + CE directional classifiers (the full
CDEv2 + CE-rerank architecture) and report test accuracy/AUC/F1.

Blend is selected on the VAL split and applied to TEST (no test tuning):
    fused_prob = sigmoid( w * logit_cde + (1-w) * logit_ce )   [logits z-scored]
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score

V2_DIR = Path(__file__).resolve().parent
KL_DIR = V2_DIR.parent
PROJECT_ROOT = KL_DIR.parent.parent
sys.path.insert(0, str(KL_DIR)); sys.path.insert(0, str(KL_DIR / "cross_encoder"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from transformers import AutoTokenizer                       # noqa: E402
from torch.utils.data import DataLoader                      # noqa: E402
import train_directional as TD                               # noqa: E402

DATA = PROJECT_ROOT / "data_6" / "aep_causal_classification"
OUT = KL_DIR / "results" / "directional"
device = torch.device("cuda")


@torch.no_grad()
def logits_for(arch, split, max_len, bs=64):
    ck = torch.load(OUT / arch / "best.pt", map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    tok = AutoTokenizer.from_pretrained(cfg["backbone"])
    model = (TD.CEClassifier(cfg["backbone"]) if arch == "ce"
             else TD.CDEDirectional(cfg["backbone"], cfg.get("warm_ckpt"))).to(device)
    model.load_state_dict(ck["model_state_dict"]); model.eval()
    coll = TD.CECollator(tok, max_len) if arch == "ce" else TD.CDECollator(tok, max_len)
    dl = DataLoader(TD.PairRows(DATA / f"directional_{split}.jsonl"), batch_size=bs,
                    shuffle=False, collate_fn=coll, num_workers=4)
    logits, labels = [], []
    for batch, y in dl:
        if arch == "ce":
            batch = {k: v.to(device) for k, v in batch.items()}
        else:
            batch = tuple((i.to(device), m.to(device)) for (i, m) in batch)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits.extend(model(batch).float().cpu().tolist())
        labels.extend(y.tolist())
    del model; torch.cuda.empty_cache()
    return np.array(logits), np.array(labels)


def z(x): return (x - x.mean()) / (x.std() + 1e-9)
def metrics(prob, y):
    pred = (prob > 0.5).astype(int)
    return {"acc": float((pred == y).mean()), "auc": float(roc_auc_score(y, prob)),
            "f1": float(f1_score(y, pred))}


def main():
    ce_max = json.loads((OUT / "ce" / "test_metrics.json").read_text())["config"]["max_length"]
    cde_max = json.loads((OUT / "cde" / "test_metrics.json").read_text())["config"]["max_length"]
    print("[fuse] scoring val + test for CE and CDE …")
    ce_v, y_v = logits_for("ce", "val", ce_max)
    cde_v, _ = logits_for("cde", "val", cde_max)
    ce_t, y_t = logits_for("ce", "test", ce_max)
    cde_t, _ = logits_for("cde", "test", cde_max)

    # select blend on VAL
    best = (-1, 0.5)
    for w in np.linspace(0, 1, 21):
        fused = 1/(1+np.exp(-(w*z(cde_v) + (1-w)*z(ce_v))))
        a = metrics(fused, y_v)["acc"]
        if a > best[0]:
            best = (a, w)
    w = best[1]
    print(f"[fuse] best val blend w(cde)={w:.2f}  val_acc={best[0]:.4f}")

    fused_t = 1/(1+np.exp(-(w*z(cde_t) + (1-w)*z(ce_t))))
    res = {
        "ce":   metrics(1/(1+np.exp(-ce_t)),  y_t),
        "cde":  metrics(1/(1+np.exp(-cde_t)), y_t),
        "fused": metrics(fused_t, y_t),
        "blend_w_cde": float(w),
        "reference": {"acc": 0.8571, "auc": 0.9031, "f1": 0.8558},
    }
    print("\n  model         acc      AUC      F1")
    for k in ("ce", "cde", "fused"):
        m = res[k]; print(f"  {k:<10} {m['acc']:.4f}  {m['auc']:.4f}  {m['f1']:.4f}")
    print(f"  {'reference':<10} 0.8571  0.9031  0.8558")
    d = (res["fused"]["acc"] - 0.8571) / 0.8571 * 100
    print(f"\n[fused vs reference]  acc {d:+.1f}%")
    (OUT / "fused_metrics.json").write_text(json.dumps(res, indent=2))
    print(f"[saved] {OUT/'fused_metrics.json'}")


if __name__ == "__main__":
    main()

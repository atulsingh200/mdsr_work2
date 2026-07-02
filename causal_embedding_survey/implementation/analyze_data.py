#!/usr/bin/env python3
"""
Data characterization for the causal-embedding survey.

Parses the AEP/AJO documentation, video-transcript, pairwise causal-classification,
workflow, and follow-up datasets and produces summary statistics + matplotlib figures.

IMPORTANT: this script only uses json / csv / collections / matplotlib. It never loads
transformers or torch (CPU OOM safety). All inputs are small JSON/JSONL files.

Run:  python3 analyze_data.py
Figures are written to ../figures/*.pdf
"""
import json
import csv
import os
import statistics
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "/mnt/localssd/automation/internship-causal-embedding/data"
FIGS = "/mnt/localssd/causal_embedding_survey/figures"
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 120,
})


def wc(path):
    with open(path) as f:
        return sum(1 for _ in f)


def read_jsonl(path, limit=None):
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# 1. Video transcripts
# ---------------------------------------------------------------------------
def analyze_transcripts():
    path = f"{DATA}/scrap_transcript/aep_transcripts.json"
    d = json.load(open(path))
    word_counts = [len(item["transcript"].split()) for item in d]

    # topic-area breakdown from the doc URL slug:
    #   .../docs/platform-learn/tutorials/<topic-area>/<page>
    topics = Counter()
    for item in d:
        parts = [p for p in item.get("url", "").split("/docs/")[-1].split("/") if p]
        topics[parts[2] if len(parts) > 2 else "other"] += 1

    stats = {
        "count": len(d),
        "words_min": min(word_counts),
        "words_max": max(word_counts),
        "words_mean": statistics.mean(word_counts),
        "words_median": statistics.median(word_counts),
        "topics": topics,
    }

    # Figure: transcript length distribution (left) + topic breakdown (right)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8),
                             gridspec_kw={"width_ratios": [1.1, 1.0]})
    axes[0].hist(word_counts, bins=30, color="#3b7dd8", edgecolor="white")
    axes[0].axvline(stats["words_median"], color="#d8413b", linestyle="--",
                    label=f"median = {stats['words_median']:.0f} words")
    axes[0].set_xlabel("Transcript length (words)")
    axes[0].set_ylabel("Number of videos")
    axes[0].set_title(f"Video transcript lengths (N={len(d)})")
    axes[0].legend()

    top = topics.most_common(12)
    labels = [t for t, _ in top][::-1]
    vals = [c for _, c in top][::-1]
    axes[1].barh(range(len(labels)), vals, color="#5aa469", edgecolor="white")
    axes[1].set_yticks(range(len(labels)))
    axes[1].set_yticklabels(labels, fontsize=8)
    axes[1].set_xlabel("Number of videos")
    axes[1].set_title("Top topic areas")
    fig.savefig(f"{FIGS}/fig_transcript_lengths.pdf", bbox_inches="tight")
    plt.close(fig)
    return stats


# ---------------------------------------------------------------------------
# 2. Pairwise causal-classification datasets
# ---------------------------------------------------------------------------
DATASETS = {
    # name: (subdir, train, val, test, has_tier)
    "aep\\_causal\\_cls34":      ("aep_causal_classification_34", "directional_train.jsonl", "directional_val.jsonl", "directional_test.jsonl", True),
    "hard\\_neg\\_semantic":     ("aep_causal_classification_hard_neg_semantic", "directional_train.jsonl", "directional_val.jsonl", "directional_test.jsonl", True),
    "ajo\\_doc\\_not\\_tier1":   ("ajo_doc_dataset_not_tier1", "directional_train.jsonl", "directional_val.jsonl", "directional_test.jsonl", True),
    "new\\_aep\\_wf\\_scrap":    ("new_aep_workflow_scrap", "directional_train.jsonl", "directional_val.jsonl", "directional_test.jsonl", False),
    "workflowllm\\_merged2":     ("workflowllm_aep_merged_2", "directional_train.jsonl", "directional_val.jsonl", "directional_test.jsonl", False),
    "ajo\\_newstyle":            ("ajo_newstyle", "train.jsonl", "val.jsonl", "test.jsonl", False),
}


def analyze_pairwise():
    rows = []
    tier_gaps = {}     # name -> list of |t1-t2|
    label1_frac = {}   # name -> fraction label==1 on train
    text_lens = {}     # name -> list of word counts (text_1) sampled
    for name, (sub, tr, va, te, has_tier) in DATASETS.items():
        base = f"{DATA}/{sub}"
        n_tr, n_va, n_te = wc(f"{base}/{tr}"), wc(f"{base}/{va}"), wc(f"{base}/{te}")
        sample = read_jsonl(f"{base}/{tr}", limit=20000)
        labels = [int(r["label"]) for r in sample if "label" in r]
        frac1 = sum(labels) / len(labels) if labels else float("nan")
        label1_frac[name] = frac1
        tl = [len(str(r.get("text_1", "")).split()) for r in sample]
        text_lens[name] = tl
        if has_tier:
            gaps = [abs(int(r["tier_1"]) - int(r["tier_2"])) for r in sample
                    if r.get("tier_1") is not None and r.get("tier_2") is not None]
            tier_gaps[name] = gaps
        rows.append({
            "name": name, "train": n_tr, "val": n_va, "test": n_te,
            "total": n_tr + n_va + n_te, "frac1": frac1,
            "text_med": statistics.median(tl), "has_tier": has_tier,
        })

    # Figure: dataset sizes (stacked train/val/test)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    names = [r["name"] for r in rows]
    x = range(len(names))
    tr = [r["train"] for r in rows]
    va = [r["val"] for r in rows]
    te = [r["test"] for r in rows]
    ax.bar(x, tr, label="train", color="#3b7dd8")
    ax.bar(x, va, bottom=tr, label="val", color="#f0a23b")
    ax.bar(x, te, bottom=[a + b for a, b in zip(tr, va)], label="test", color="#d8413b")
    ax.set_xticks(list(x))
    ax.set_xticklabels([n.replace("\\\\", "") for n in names], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Number of pairs")
    ax.set_title("Pairwise causal-classification dataset sizes")
    ax.legend()
    fig.savefig(f"{FIGS}/fig_dataset_sizes.pdf", bbox_inches="tight")
    plt.close(fig)

    # Figure: label balance
    fig, ax = plt.subplots(figsize=(7, 3.4))
    fr = [r["frac1"] for r in rows]
    ax.bar(x, fr, color="#5aa469", edgecolor="white")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, label="balanced (0.5)")
    ax.set_ylim(0, 1)
    ax.set_xticks(list(x))
    ax.set_xticklabels([n.replace("\\\\", "") for n in names], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Fraction of label = 1\n(text$_1$ precedes text$_2$)")
    ax.set_title("Label balance across pairwise datasets (train)")
    ax.legend()
    fig.savefig(f"{FIGS}/fig_label_balance.pdf", bbox_inches="tight")
    plt.close(fig)

    # Figure: tier-gap distribution (only tiered datasets)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    maxgap = 0
    for name, gaps in tier_gaps.items():
        maxgap = max(maxgap, max(gaps) if gaps else 0)
    bins = range(0, maxgap + 2)
    colors = ["#3b7dd8", "#d8413b", "#5aa469"]
    for i, (name, gaps) in enumerate(tier_gaps.items()):
        c = Counter(gaps)
        tot = sum(c.values())
        xs = list(range(0, maxgap + 1))
        ys = [c.get(g, 0) / tot for g in xs]
        ax.plot(xs, ys, marker="o", label=name.replace("\\\\", ""), color=colors[i % len(colors)])
    ax.set_xlabel(r"Tier gap $|\,$tier$_1$ $-$ tier$_2|$")
    ax.set_ylabel("Fraction of training pairs")
    ax.set_title("Tier-gap distribution (training pairs)")
    ax.legend()
    fig.savefig(f"{FIGS}/fig_tier_gap_dist.pdf", bbox_inches="tight")
    plt.close(fig)

    return rows, tier_gaps, text_lens


# ---------------------------------------------------------------------------
# 3. Follow-up queries
# ---------------------------------------------------------------------------
def analyze_followups():
    path = "/mnt/localssd/z_followupq/prod_jan_feb_26-aep-ajo_follow-up-queries.json"
    d = json.load(open(path))

    def n_followups(item):
        fq = item["follow_up_queries"]
        if fq and isinstance(fq[0], list):
            return sum(len(s) for s in fq)
        return len(fq)

    counts = [n_followups(x) for x in d]
    qlens = [len(x["question"].split()) for x in d]
    stats = {
        "count": len(d),
        "fu_min": min(counts), "fu_max": max(counts),
        "fu_mean": statistics.mean(counts), "fu_median": statistics.median(counts),
        "qlen_mean": statistics.mean(qlens),
        "dist": Counter(counts),
    }

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))
    c = stats["dist"]
    xs = sorted(c)
    axes[0].bar(xs, [c[k] for k in xs], color="#7b5ad8", edgecolor="white")
    axes[0].set_xlabel("# follow-up queries per question")
    axes[0].set_ylabel("Number of questions")
    axes[0].set_title(f"Follow-ups per question (N={len(d)})")
    axes[1].hist(qlens, bins=25, color="#3b9dd8", edgecolor="white")
    axes[1].set_xlabel("Anchor question length (words)")
    axes[1].set_ylabel("Number of questions")
    axes[1].set_title("Anchor question length")
    fig.savefig(f"{FIGS}/fig_followup_counts.pdf", bbox_inches="tight")
    plt.close(fig)
    return stats


# ---------------------------------------------------------------------------
# 4. Workflows (orchestrated + 3-step)
# ---------------------------------------------------------------------------
def analyze_workflows():
    flat = json.load(open("/mnt/localssd/ajo_orchestrated_workflows_flat.json"))
    n_steps_flat = []
    for w in flat:
        steps = [k for k in w if k.startswith("t") and k[1:].isdigit()]
        n_steps_flat.append(len(steps))
    out = {"ajo_orchestrated_n": len(flat), "ajo_steps": Counter(n_steps_flat)}
    for f in ["aep_3step_workflows", "aep_3step_workflows_simple",
              "aep_3step_workflows_specific", "aep_3step_workflows_transcript"]:
        p = f"{DATA}/scrap_transcript/{f}.json"
        out[f] = len(json.load(open(p)))
    return out


def main():
    print("== transcripts ==")
    tr = analyze_transcripts()
    for k, v in tr.items():
        if k != "topics":
            print(f"  {k}: {v}")
    print("  topics:", dict(tr["topics"].most_common()))

    print("== pairwise ==")
    rows, gaps, tlens = analyze_pairwise()
    for r in rows:
        print(f"  {r['name']:22s} train={r['train']:>7d} val={r['val']:>6d} "
              f"test={r['test']:>6d} frac1={r['frac1']:.3f} text_med={r['text_med']:.0f} tier={r['has_tier']}")
    for name, g in gaps.items():
        adj = sum(1 for x in g if x == 0) / len(g)
        print(f"  tier-gap {name}: adjacent(gap0)={adj:.3f} max={max(g)}")

    print("== followups ==")
    fu = analyze_followups()
    for k, v in fu.items():
        if k != "dist":
            print(f"  {k}: {v}")

    print("== workflows ==")
    wf = analyze_workflows()
    for k, v in wf.items():
        print(f"  {k}: {v}")

    print("\n== figures ==")
    for fn in ["fig_transcript_lengths.pdf", "fig_dataset_sizes.pdf",
               "fig_label_balance.pdf", "fig_tier_gap_dist.pdf",
               "fig_followup_counts.pdf"]:
        p = f"{FIGS}/{fn}"
        sz = os.path.getsize(p) if os.path.exists(p) else 0
        print(f"  {fn}: {sz} bytes {'OK' if sz > 0 else 'MISSING'}")


if __name__ == "__main__":
    main()

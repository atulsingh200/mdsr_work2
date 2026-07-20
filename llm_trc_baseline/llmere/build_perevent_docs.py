"""LLMERE Phase B — build per-event O(n) extraction data from final_data (self-contained).

Reconstructs workflow "documents" by grouping the pairwise final_data on `url`, using the tier index
(`step_i` = 'tN') as the gold total order. Each document's events are tagged <e0 ...> <e1 ...> in a
SHUFFLED presentation order (so the tag index does NOT leak gold order). For a specified event, the
task is to enumerate ALL events it precedes ("AFTER"), turning the pairwise O(n^2) task into per-event
O(n). Training targets include a transitive-chain rationale (LLMERE §3.3.2).

Outputs (JSONL) under outputs/llmere/perevent/:
  train.jsonl / test.jsonl  — one row per (document, specified-event) query:
    {doc_id, n_events, events:[{tag,text,gold_rank}], query_tag, gold_after:[tags],
     rationale, prompt, target}

Run:
  .venv/bin/python llmere/build_perevent_docs.py --neg-ratio 1.0 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

CER = Path("/mnt/localssd/causal-embedding-research")
SPLITS = {"train": CER / "final_data/directional_train.jsonl",
          "test": CER / "final_data/directional_test.jsonl"}
HERE = Path(__file__).parent
OUT = HERE.parent / "outputs/llmere/perevent"

TASK_DESC = (
    "The current task is a procedural step-ordering extraction task. You are given the steps of a "
    "procedure, each marked as <eK text>. Steps may be listed out of order. For the specified step, "
    "identify ALL other steps that must happen AFTER it in the correct procedure. First give the "
    "answer line 'AFTER: <e..>, <e..>' (or 'AFTER: none'), then a 'Reasoning:' line with the "
    "transitive ordering chain."
)


def _tier(v):
    if v is None:
        return None
    s = "".join(c for c in str(v) if c.isdigit())
    return int(s) if s else None


def reconstruct(path):
    """url -> {tier: text}. One doc per url; gold order = ascending tier."""
    docs = defaultdict(dict)
    for ln in open(path):
        ln = ln.strip()
        if not ln:
            continue
        r = json.loads(ln)
        for t, txt in [(r.get("step_1"), r.get("text_1")), (r.get("step_2"), r.get("text_2"))]:
            ti = _tier(t)
            if ti is not None and txt:
                docs[r["url"]].setdefault(ti, txt)
    return docs


def build_examples(docs, neg_ratio, seed, min_events=3, max_events=30):
    rng = random.Random(seed)
    rows = []
    for url, tier2txt in docs.items():
        tiers = sorted(tier2txt)                       # ascending gold order
        if not (min_events <= len(tiers) <= max_events):
            continue
        # gold rank = position in ascending tier order (0-based)
        ordered = [(rank, tier2txt[t]) for rank, t in enumerate(tiers)]
        # shuffle presentation order and assign event tags <e0..> in that shuffled order
        perm = ordered[:]
        rng.shuffle(perm)
        tag_of_rank = {}                               # gold_rank -> event tag index
        events = []
        for tag_idx, (gold_rank, text) in enumerate(perm):
            tag_of_rank[gold_rank] = tag_idx
            events.append({"tag": tag_idx, "text": text, "gold_rank": gold_rank})
        doc_str = " ".join(f"<e{e['tag']} {e['text']}>" for e in events)

        # one query per event (O(n)); positives have >=1 successor, negatives (last event) => none
        for gold_rank in range(len(tiers)):
            q_tag = tag_of_rank[gold_rank]
            after_ranks = list(range(gold_rank + 1, len(tiers)))
            after_tags = [tag_of_rank[r] for r in after_ranks]
            is_neg = len(after_tags) == 0
            if is_neg and rng.random() > neg_ratio:    # subsample negatives
                continue
            # transitive-chain rationale: gold_rank -> gold_rank+1 -> ... (in tag space)
            chain = " ; ".join(
                f"<e{tag_of_rank[r]}> before <e{tag_of_rank[r+1]}>" for r in after_ranks[:-1]
            ) if len(after_ranks) >= 2 else (
                f"<e{q_tag}> before <e{after_tags[0]}>" if after_tags else "no later steps")
            ans = ("AFTER: " + ", ".join(f"<e{t}>" for t in sorted(after_tags))) if after_tags else "AFTER: none"
            target = f"{ans}\nReasoning: {chain}"
            instruction = (f"Please identify all steps that must happen AFTER the given step "
                           f"<e{q_tag} {events_text(events, q_tag)}>.")
            prompt = f"{TASK_DESC}\n\nDocument:\n{doc_str}\n\nInstruction:\n{instruction}"
            rows.append({
                "doc_id": url, "n_events": len(tiers),
                "query_tag": q_tag, "query_gold_rank": gold_rank,
                "gold_after_tags": sorted(after_tags),
                "events": events, "prompt": prompt, "target": target,
            })
    return rows


def events_text(events, tag):
    for e in events:
        if e["tag"] == tag:
            return e["text"]
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neg-ratio", type=float, default=1.0, help="fraction of terminal (none) queries to keep")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-events", type=int, default=3)
    ap.add_argument("--max-events", type=int, default=30)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for split, path in SPLITS.items():
        docs = reconstruct(path)
        rows = build_examples(docs, args.neg_ratio, args.seed, args.min_events, args.max_events)
        (OUT / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        n_docs = len({r["doc_id"] for r in rows})
        n_pos = sum(1 for r in rows if r["gold_after_tags"])
        print(f"{split}: docs={n_docs}  queries={len(rows)}  positive={n_pos}  "
              f"negative(none)={len(rows)-n_pos}  -> {OUT/f'{split}.jsonl'}")


if __name__ == "__main__":
    main()

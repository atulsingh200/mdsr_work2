#!/usr/bin/env python3
"""
Strictly serial Claude Sonnet 4.6 eval with sanity-check on each response.
Detects degraded "all-0" periods and waits them out.
Saves incrementally every 50 rows.
"""
from __future__ import annotations
import json, time, urllib.request, urllib.error
from pathlib import Path
from collections import deque
from sklearn.metrics import f1_score

TEST_PATH  = Path("/mnt/localssd/causal-embedding-research/final_data/directional_test.jsonl")
OUT_PATH   = Path("/mnt/localssd/causal-embedding-research/final_data/claude_sonnet46_test_results.json")
CKPT_PATH  = Path("/mnt/localssd/causal-embedding-research/final_data/claude_sonnet46_checkpoint.jsonl")

API_URL = "https://mdsr-foundry-resource.services.ai.azure.com/anthropic/v1/messages"
API_KEY = "5rTAAMibReGzw2jVYvjoAiK7NE9hYoTqT8vtT8Wu1vsLHs9HUpnpJQQJ99CFACHYHv6XJ3w3AAAAACOGU5x9"
MODEL   = "claude-sonnet-4-6"

ICL_EXAMPLES = [
    {"text_1": "Open the rule editor and create a new rule for the page load event.",
     "text_2": "Add the Data Variable data element to the Data field of the rule action.",
     "label": 1},
    {"text_1": "Click Save to store the configuration settings.",
     "text_2": "In the left navigation panel, select Extensions.",
     "label": 0},
]
SYSTEM_PROMPT = (
    "You are an expert at understanding procedural workflows for Adobe Experience Platform (AEP) "
    "and Adobe Journey Optimizer (AJO).\n\n"
    "Given two workflow steps (Step A and Step B), decide whether Step A causally precedes "
    "Step B — i.e., Step A must happen BEFORE Step B in the correct workflow order.\n\n"
    "Reply with exactly ONE digit:\n1 = Step A happens before Step B\n0 = Step A does NOT happen before Step B\n\nNo explanation."
)

def build_user(t1, t2):
    ex = ""
    for i, e in enumerate(ICL_EXAMPLES, 1):
        ex += f"Example {i}:\nStep A: {e['text_1']}\nStep B: {e['text_2']}\nAnswer: {e['label']}\n\n"
    return f"{ex}Now answer:\nStep A: {t1}\nStep B: {t2}\nAnswer:"

def call_once(t1, t2):
    data = json.dumps({"model": MODEL, "max_tokens": 5, "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": build_user(t1, t2)}]}).encode()
    req = urllib.request.Request(API_URL, data=data, headers={
        "Content-Type": "application/json", "x-api-key": API_KEY, "anthropic-version": "2023-06-01"})
    resp = urllib.request.urlopen(req, timeout=60)
    return json.loads(resp.read())["content"][0]["text"].strip()

def call_with_retry(t1, t2):
    for attempt in range(8):
        try:
            raw = call_once(t1, t2)
            return raw
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            wait = min(60, 4 ** attempt)
            if e.code in (429, 529) or "overloaded" in body.lower():
                print(f"  [rate limit {e.code}] sleeping {wait}s", flush=True)
                time.sleep(wait)
            else:
                raise
        except Exception as e:
            wait = min(30, 2 ** attempt)
            print(f"  [err attempt {attempt}: {e}] sleeping {wait}s", flush=True)
            time.sleep(wait)
    return ""

def parse(raw):
    t = raw.strip()
    if t.startswith("1"): return 1
    if t.startswith("0"): return 0
    return None

def main():
    rows = [json.loads(l) for l in open(TEST_PATH)]
    print(f"Total: {len(rows)}")

    done = {}
    if CKPT_PATH.exists():
        for line in CKPT_PATH.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["idx"]] = r
    print(f"Checkpoint: {len(done)} done, {len(rows)-len(done)} remaining")

    ckpt_f = open(CKPT_PATH, "a")
    recent_preds = deque(maxlen=20)  # rolling window for degradation detection
    t0 = time.time()

    for idx, row in enumerate(rows):
        if idx in done:
            continue

        # Degrade detection: if last 20 preds are all 0, pause 30s
        if len(recent_preds) == 20 and all(p == 0 for p in recent_preds):
            print(f"  [DEGRADED at idx={idx}] all-zero window detected, pausing 30s...", flush=True)
            time.sleep(30)
            recent_preds.clear()

        raw = call_with_retry(row["text_1"], row["text_2"])
        pred = parse(raw)
        if pred is None:
            pred = 0

        rec = {"idx": idx, "pred": pred, "label": int(row["label"]),
               "raw": raw, "source": row.get("source", "")}
        done[idx] = rec
        recent_preds.append(pred)
        ckpt_f.write(json.dumps(rec) + "\n")
        ckpt_f.flush()

        if (idx + 1) % 100 == 0:
            all_done = list(done.values())
            acc = sum(r["pred"]==r["label"] for r in all_done) / len(all_done)
            p0  = sum(1 for r in all_done[-200:] if r["pred"]==0) / min(200, len(all_done))
            elapsed = time.time() - t0
            rate = len([r for r in all_done if r["idx"] >= min(done.keys(), default=0)]) / elapsed
            eta  = (len(rows) - idx - 1) / max(rate, 0.1)
            print(f"  [{idx+1}/{len(rows)}]  acc={acc:.4f}  recent_p0={p0:.2f}  {rate:.1f} req/s  ETA={eta/60:.1f}min", flush=True)

        time.sleep(0.25)  # 4 req/s max

    ckpt_f.close()

    all_results = [done[i] for i in range(len(rows))]
    preds  = [r["pred"]  for r in all_results]
    labels = [r["label"] for r in all_results]
    acc  = sum(p==l for p,l in zip(preds,labels)) / len(labels)
    f1   = f1_score(labels, preds)
    f1_0 = f1_score(labels, preds, pos_label=0)
    by_src = {}
    for r in all_results:
        s = r["source"]
        by_src.setdefault(s, {"correct":0,"total":0})
        by_src[s]["total"] += 1
        if r["pred"]==r["label"]: by_src[s]["correct"] += 1

    print(f"\n{'='*55}")
    print(f"CLAUDE SONNET 4.6 — FINAL RESULTS")
    print(f"{'='*55}")
    print(f"  n        : {len(labels)}")
    print(f"  acc      : {acc:.4f}")
    print(f"  f1 (1)   : {f1:.4f}")
    print(f"  f1 (0)   : {f1_0:.4f}")
    print(f"  By source:")
    for s,d in sorted(by_src.items()):
        print(f"    {s:<12}  acc={d['correct']/d['total']:.4f}  n={d['total']}")

    output = {"model":MODEL,"n":len(labels),"shots":2,"acc":round(acc,4),
              "f1":round(f1,4),"f1_0":round(f1_0,4),
              "by_source":{s:{"acc":round(d["correct"]/d["total"],4),"n":d["total"]} for s,d in by_src.items()}}
    OUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nSaved -> {OUT_PATH}")

if __name__ == "__main__":
    main()

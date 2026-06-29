#!/usr/bin/env python3
"""
Build EXECUTION-ORDER (workflow) pairs from AJO tutorial transcripts.

Principle (per user decision):
  - Order signal = position within a single transcript (earlier sentence = earlier step).
  - Pairs are SAME-DOCUMENT only -> no cross-topic / tier conflict.
  - label = 1  means text_1 is executed BEFORE text_2 (forward).
  - label = 0  is the reversed pair (text_2 before text_1).
  - Both directions emitted (antisymmetric supervision), so the head learns a
    real comparator instead of a register/topic shortcut.

Schema matches directional_*.jsonl exactly:
  text_1, text_2, label, tier_1, tier_2, sub_1, sub_2, url_1, url_2
  (tier_*/sub_* carried from the doc for compatibility; both sides share the
   same doc here, so tier_1==tier_2 and sub_1==sub_2.)
"""
import json, re, random, sys
from pathlib import Path
from collections import defaultdict

# ---- reuse the tier table + url->subchapter resolution from the real build file
sys.path.insert(0, "/mnt/user-data/uploads")
import importlib.util
spec = importlib.util.spec_from_file_location(
    "bld", "/mnt/user-data/uploads/build_classification_data__3_.py")
bld = importlib.util.module_from_spec(spec)
# the module's main() won't run on import; we only want its tables/helpers
spec.loader.exec_module(bld)

CORPUS = "/mnt/user-data/uploads/aep_docs_collection_v_24-03-2026.json"
OUTDIR = Path("/home/claude/data_workflow_order"); OUTDIR.mkdir(exist_ok=True)

# ---------- text extraction (same field path the build script uses) ----------
def _urlfrac(t):
    u = sum(len(m.group()) for m in re.finditer(r"https?://\S+", t))
    return u / max(len(t), 1)

def load_docs():
    data = json.load(open(CORPUS))
    docs = {}
    for g in data:
        if not g:
            continue
        url = g[0]["metadata"].get("sourceUrl")
        if not url:
            continue
        title = g[0]["metadata"].get("title", "")
        texts = []
        for e in g:
            for ch in e.get("chunks", []) or []:
                t = ch.get("data") or ch.get("text") or ch.get("content") or ""
                if t and _urlfrac(t) <= 0.3:
                    texts.append(t)
        body = "\n".join(texts).strip()
        if body:
            docs[url] = {"title": title, "text": body}
    return docs

# ---------- transcript / step detection ----------
# Keep only genuine procedural transcripts (the "learn/tutorials" videos),
# and within them, only ACTION sentences in transcript order.
ACTION_VERBS = (
    "select","click","choose","navigate","go to","open","create","add","enter",
    "type","drag","drop","name","save","publish","configure","set","define",
    "map","ingest","upload","import","export","build","enable","toggle","check",
    "fill","specify","activate","send","launch","review","preview","test","apply",
    "assign","switch","search","scroll","hit","press","pick","provide","confirm",
)
ACTION_RE = re.compile(r"\b(" + "|".join(v.replace(" ", r"\s") for v in ACTION_VERBS) + r")\b", re.I)
# sentences that are conceptual enumeration / narration to drop
DROP_RE = re.compile(
    r"\b(in this video|let me|let's first look|we call this|the framework|"
    r"thanks for watching|recommendation-more-help|on one hand|on the other hand|"
    r"challenges facing|overview of)\b", re.I)
HELP_RE = re.compile(r"recommendation-more-help.*", re.S)

def is_tutorial(url):
    return "/tutorials/" in url

def sentences(text):
    text = HELP_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    # split on sentence boundaries
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]

def action_steps(text, min_words=5, max_words=60):
    out = []
    for s in sentences(text):
        wc = len(s.split())
        if wc < min_words or wc > max_words:
            continue
        if DROP_RE.search(s):
            continue
        if not ACTION_RE.search(s):
            continue
        out.append(s)
    return out

# ---------- url -> (sub, tier) using the real resolver tables ----------
def resolve_sub_tier(url):
    # The build script resolves via TOC; we approximate using its slug rules.
    # Fall back to scanning CHAPTER_RULES for a slug match.
    slug = url.rstrip("/").split("/")[-1]
    for chap, rule in bld.CHAPTER_RULES.items():
        if "by_slug" in rule and slug in rule["by_slug"]:
            sub = rule["by_slug"][slug]
            return sub, bld.SUBCHAPTER_TO_TIER.get(sub)
        if "by_direct_slug" in rule and slug in rule["by_direct_slug"]:
            sub = rule["by_direct_slug"][slug]
            return sub, bld.SUBCHAPTER_TO_TIER.get(sub)
    return None, None

# ---------- build pairs ----------
def main():
    rng = random.Random(42)
    docs = load_docs()
    tutorials = {u: d for u, d in docs.items() if is_tutorial(u)}
    print(f"[load] {len(docs)} docs, {len(tutorials)} tutorials", file=sys.stderr)

    # gather per-doc ordered steps
    doc_steps = {}
    for url, d in tutorials.items():
        steps = action_steps(d["text"])
        if len(steps) >= 3:                      # need a real sequence
            sub, tier = resolve_sub_tier(url)
            doc_steps[url] = {"steps": steps, "sub": sub, "tier": tier}
    print(f"[steps] {len(doc_steps)} tutorials with >=3 action steps", file=sys.stderr)

    # split by DOCUMENT (no leakage): 80/10/10 over docs
    urls = list(doc_steps); rng.shuffle(urls)
    n = len(urls); ntest = int(0.1*n); nval = int(0.1*n)
    split = {}
    for i, u in enumerate(urls):
        split[u] = "test" if i < ntest else "val" if i < ntest+nval else "train"

    MAX_GAP = 4        # pair steps within a window so far-apart unrelated lines don't dominate
    PER_DOC_CAP = 60   # cap pairs per doc so long transcripts don't swamp

    rows = {"train": [], "val": [], "test": []}
    for url, info in doc_steps.items():
        steps, sub, tier = info["steps"], info["sub"], info["tier"]
        sp = split[url]
        pairs = []
        for i in range(len(steps)):
            for j in range(i+1, min(i+1+MAX_GAP, len(steps))):
                a, b = steps[i], steps[j]
                if a == b:
                    continue
                pairs.append((a, b))
        rng.shuffle(pairs)
        pairs = pairs[:PER_DOC_CAP]
        for a, b in pairs:
            base = dict(tier_1=tier, tier_2=tier, sub_1=sub, sub_2=sub,
                        url_1=url, url_2=url)
            # forward
            rows[sp].append({"text_1": a, "text_2": b, "label": 1, **base})
            # reverse
            rows[sp].append({"text_1": b, "text_2": a, "label": 0, **base})

    for sp in ("train", "val", "test"):
        rng.shuffle(rows[sp])
        with open(OUTDIR / f"directional_{sp}.jsonl", "w") as f:
            for r in rows[sp]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts = {sp: len(rows[sp]) for sp in rows}
    lbl = defaultdict(int)
    for r in rows["train"]:
        lbl[r["label"]] += 1
    print(f"[done] rows: {counts}", file=sys.stderr)
    print(f"[done] train label balance: {dict(lbl)}", file=sys.stderr)
    print(f"[done] docs/split: "
          f"{sum(1 for v in split.values() if v=='train')} train / "
          f"{sum(1 for v in split.values() if v=='val')} val / "
          f"{sum(1 for v in split.values() if v=='test')} test", file=sys.stderr)

if __name__ == "__main__":
    main()

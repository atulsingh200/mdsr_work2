#!/usr/bin/env python3
"""
Build a UNIFIED CAUSAL-PRECEDENCE dataset (binary) from the AJO corpus.

ONE label meaning, two scales:
    label = 1  ->  text_1 causally PRECEDES text_2  (A is a prerequisite of B)
    label = 0  ->  everything else:
                     - the reverse of a precedence pair, OR
                     - no-relation pairs (unordered / incomparable)

Two sources of "precedence":
    (A) WORKFLOW (micro): within a single tutorial transcript, an earlier action
        step precedes a later one. Position in transcript = execution order.
    (B) CURRICULUM (macro): across topics, a prerequisite topic precedes a
        dependent topic, as defined by the PREREQ_DAG below (NOT tier rank).
        Includes transitive (skip) edges so the model learns transitivity.

Negatives (label 0):
    - reverse of every positive (antisymmetry), AND
    - NO-RELATION pairs: sibling workflow steps far apart? no -> still ordered.
      For curriculum: pairs of topics with NO path between them in the DAG.
      For workflow: cross-document step pairs (different procedures => no causal
      order) sampled as no-relation negatives.

Output schema (identical to your directional_*.jsonl):
    text_1, text_2, label, tier_1, tier_2, sub_1, sub_2, url_1, url_2
A `scale` field is appended too (workflow|curriculum|norel); drop it if your
loader rejects unknown keys (it won't if it does dict access by key).

USAGE:
    python build_causal_dataset.py --corpus path/to/aep_docs_collection.json \
        --build-script path/to/build_classification_data.py \
        --out-dir data/aep_causal --seed 42
"""
import argparse, json, re, random, sys, importlib.util
from pathlib import Path
from collections import defaultdict, deque

# ===========================================================================
# 1) PREREQUISITE DAG  --  EDIT THIS.  edge "X": ["Y", ...]  means X precedes Y
#    (X is a prerequisite of Y).  Nodes are subchapter codes from your table.
#    Drafted from your 15-tier structure but expressed as REAL dependencies,
#    not linear tier rank.  Review/correct the edges.
# ===========================================================================
PREREQ_DAG: dict[str, list[str]] = {
    # Orientation precedes everything conceptual
    "2a": ["17a", "14a", "8a"],
    "2b": ["9a"],                       # mobile orientation -> channels
    # Admin floor precedes data + config
    "17a": ["14a", "16a"],
    "17c": ["14a"],
    # Data foundations -> profiles/audiences
    "14a": ["14b", "8a"],              # schema -> dataset -> profiles
    "14b": ["14c", "8a"],
    "14c": ["8a"],
    # Profiles -> audiences -> subscriptions
    "8a": ["8b", "10a"],
    "8b": ["8c", "9a"],                # audiences -> channels
    "8c": ["9a"],
    # Content authoring -> channels
    "10a": ["10b", "9a"], "10b": ["10c"], "10c": ["10d"], "10d": ["10e"],
    "10e": ["10f"], "10f": ["10g"], "10g": ["9a"],
    # Channels/campaigns -> journeys
    "9a": ["9b", "5a"], "9b": ["9c"], "9c": ["9d"], "9d": ["9e"], "9e": ["5a"],
    "4a": ["5a"], "4b": ["5a"], "4c": ["5a"],
    # Journeys core -> advanced -> orchestration
    "5a": ["5b", "11a"], "5b": ["5c", "12a"],
    "5c": ["7a"], "12a": ["12b"], "12b": ["7a"],
    "11a": ["11b"], "11b": ["7a"],
    "13a": ["13b"], "13b": ["13c"], "13c": ["7a"],
    "7a": ["7b"], "7b": ["15a"],
    # Admin config / governance / reporting feed capstones
    "16a": ["16b"], "16b": ["16c"], "16c": ["15a"], "17b": ["15a"],
    "18a": ["18b"], "18b": ["18c"], "18c": ["15a"],
    "15a": ["15b"], "15b": ["15c"], "15c": ["15d"], "15d": ["15e"], "15e": ["6"],
    # AI assistants -> capstones
    "19a": ["19b"], "19b": ["6"], "2c": ["6"],
    # Capstones / use cases are terminal
    "6": ["20a"], "20a": ["20b"], "20b": ["20c"], "20c": ["20d"],
    "21a": ["21b"], "21b": ["22"],
}

# ===========================================================================
# 2) corpus extraction (matches your build script's field path)
# ===========================================================================
def urlfrac(t: str) -> float:
    u = sum(len(m.group()) for m in re.finditer(r"https?://\S+", t))
    return u / max(len(t), 1)

def load_docs(corpus_path: str):
    data = json.load(open(corpus_path))
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
                if t and urlfrac(t) <= 0.3:
                    texts.append(t)
        body = "\n".join(texts).strip()
        if body:
            # keep longer body on duplicate url
            if url not in docs or len(body) > len(docs[url]["text"]):
                docs[url] = {"title": title, "text": body}
    return docs

# ===========================================================================
# 3) sentence + action-step extraction (workflow source)
# ===========================================================================
ACTION_VERBS = ("select","click","choose","navigate","go to","open","create","add",
    "enter","type","drag","drop","name","save","publish","configure","set","define",
    "map","ingest","upload","import","export","build","enable","toggle","check","fill",
    "specify","activate","send","launch","review","preview","test","apply","assign",
    "switch","search","scroll","hit","press","pick","provide","confirm")
ACTION_RE = re.compile(r"\b(" + "|".join(v.replace(" ", r"\s") for v in ACTION_VERBS) + r")\b", re.I)
DROP_RE = re.compile(r"\b(in this video|thanks for watching|recommendation-more-help|"
    r"let me give you|overview of the|we call this|on one hand|on the other hand)\b", re.I)
HELP_RE = re.compile(r"recommendation-more-help.*", re.S)

def sentences(text: str):
    text = HELP_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]

def windows(sents, lo=2, hi=5):
    """Group consecutive sentences into 2-5 sentence units, preserving order."""
    out, i = [], 0
    n = len(sents)
    while i < n:
        size = min(hi, max(lo, 3), n - i)
        out.append(" ".join(sents[i:i+size]))
        i += size
    return out

def action_steps(text, min_words=5, max_words=80):
    keep = []
    for s in sentences(text):
        wc = len(s.split())
        if wc < min_words or wc > max_words: continue
        if DROP_RE.search(s): continue
        if not ACTION_RE.search(s): continue
        keep.append(s)
    return keep

# ===========================================================================
# 4) subchapter / tier resolution via your real build script tables
# ===========================================================================
def load_build_tables(build_script_path: str):
    spec = importlib.util.spec_from_file_location("bld", build_script_path)
    bld = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bld)
    return bld

def resolve_sub_tier(url, bld):
    slug = url.rstrip("/").split("/")[-1]
    for chap, rule in bld.CHAPTER_RULES.items():
        for key in ("by_slug", "by_direct_slug"):
            if key in rule and slug in rule[key]:
                sub = rule[key][slug]
                return sub, bld.SUBCHAPTER_TO_TIER.get(sub)
    return None, None

# ===========================================================================
# 5) DAG helpers: ancestors (transitive prerequisites) + comparability
# ===========================================================================
def transitive_closure(dag):
    nodes = set(dag) | {y for ys in dag.values() for y in ys}
    succ = {n: set() for n in nodes}
    for x, ys in dag.items():
        for y in ys:
            succ[x].add(y)
    # BFS closure
    closure = {n: set() for n in nodes}
    for n in nodes:
        seen, q = set(), deque(succ[n])
        while q:
            m = q.popleft()
            if m in seen: continue
            seen.add(m)
            q.extend(succ.get(m, ()))
        closure[n] = seen
    return nodes, closure

# ===========================================================================
# 6) main build
# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--build-script", default="build_classification_data.py",
                    help="path to your build_classification_data.py (for tier tables)")
    ap.add_argument("--out-dir", default="data/aep_causal")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--unit", choices=["sentence", "window"], default="window",
                    help="window = 2-5 sentence units (matches your original)")
    ap.add_argument("--max-workflow-gap", type=int, default=4)
    ap.add_argument("--per-doc-cap", type=int, default=80)
    ap.add_argument("--max-curric-pairs-per-edge", type=int, default=120)
    ap.add_argument("--workflow-repeat", type=int, default=4,
                    help="duplicate workflow pairs N times to balance vs curriculum")
    ap.add_argument("--norel-ratio", type=float, default=0.5,
                    help="no-relation negatives as fraction of positives")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    bld = load_build_tables(args.build_script)
    docs = load_docs(args.corpus)
    print(f"[load] {len(docs)} docs", file=sys.stderr)

    # ---- attach sub/tier + extract units per doc
    by_sub = defaultdict(list)          # sub -> list of (url, tier, unit_text)
    tutorial_steps = {}                 # url -> {steps, sub, tier}
    for url, d in docs.items():
        sub, tier = resolve_sub_tier(url, bld)
        if sub is None:
            continue
        # curriculum units: 2-5 sentence windows of the whole doc
        sents = sentences(d["text"])
        units = windows(sents) if args.unit == "window" else sents
        for u in units:
            if 8 <= len(u.split()):
                by_sub[sub].append((url, tier, u))
        # workflow units: ordered action steps (tutorials only)
        if "/tutorials/" in url:
            steps = action_steps(d["text"])
            if len(steps) >= 3:
                tutorial_steps[url] = {"steps": steps, "sub": sub, "tier": tier}
    print(f"[units] subs={len(by_sub)}  tutorials_with_steps={len(tutorial_steps)}",
          file=sys.stderr)

    # ---- URL-level split (80/10/10, no leakage): each document URL is assigned
    #      to exactly one split. Pairs are emitted into the split of their
    #      *source* URL (workflow) or the shared split when both URLs agree
    #      (curriculum). Cross-split curriculum pairs are dropped to avoid leakage.
    all_urls = sorted({url for url, _, _ in (u for units in by_sub.values() for u in units)}
                      | set(tutorial_steps.keys()))
    rng.shuffle(all_urls)
    nu = len(all_urls)
    n_test = max(1, int(0.1 * nu))
    n_val  = max(1, int(0.1 * nu))
    url_split = {}
    for i, u in enumerate(all_urls):
        url_split[u] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    print(f"[split] {nu} URLs -> train={nu-n_test-n_val}  val={n_val}  test={n_test}",
          file=sys.stderr)
    def url_to_split_via_sub(sub):
        # fallback used only for curriculum pairs — resolved per-URL below
        return "train"

    rows = {"train": [], "val": [], "test": []}
    def emit(split, a_url, a_sub, a_tier, a_txt, b_url, b_sub, b_tier, b_txt,
             label, scale):
        rows[split].append({
            "text_1": a_txt, "text_2": b_txt, "label": label,
            "tier_1": a_tier, "tier_2": b_tier, "sub_1": a_sub, "sub_2": b_sub,
            "url_1": a_url, "url_2": b_url, "scale": scale,
        })

    # ====================== SOURCE A: workflow precedence ===================
    wf_pos = 0
    for url, info in tutorial_steps.items():
        steps, sub, tier = info["steps"], info["sub"], info["tier"]
        sp = url_to_split_via_sub(sub)
        pairs = []
        for i in range(len(steps)):
            for j in range(i+1, min(i+1+args.max_workflow_gap, len(steps))):
                if steps[i] != steps[j]:
                    pairs.append((steps[i], steps[j]))
        rng.shuffle(pairs); pairs = pairs[:args.per_doc_cap]
        for a, b in pairs:
            for _ in range(args.workflow_repeat):
                emit(sp, url, sub, tier, a, url, sub, tier, b, 1, "workflow")   # before
                emit(sp, url, sub, tier, b, url, sub, tier, a, 0, "workflow")   # reverse
            wf_pos += 1
    print(f"[workflow] {wf_pos} positive step-pairs (x2 with reverse)", file=sys.stderr)

    # ====================== SOURCE B: curriculum precedence =================
    nodes, closure = transitive_closure(PREREQ_DAG)
    # positive curriculum edges = (X, Y) where Y in closure[X]  (X prereq of Y)
    cur_pos = 0
    for x in nodes:
        for y in closure[x]:
            if x not in by_sub or y not in by_sub:
                continue
            # both topics must be in the same split (guaranteed for same sub-split)
            if sub_split.get(x) != sub_split.get(y):
                continue
            sp = sub_split.get(x, "train")
            ax, ay = by_sub[x], by_sub[y]
            k = min(args.max_curric_pairs_per_edge, len(ax)*len(ay))
            for _ in range(k):
                ua, ta, txa = rng.choice(ax)
                ub, tb, txb = rng.choice(ay)
                emit(sp, ua, x, ta, txa, ub, y, tb, txb, 1, "curriculum")   # prereq before
                emit(sp, ub, y, tb, txb, ua, x, ta, txa, 0, "curriculum")   # reverse
                cur_pos += 1
    print(f"[curriculum] {cur_pos} positive prereq-pairs (x2 with reverse)", file=sys.stderr)

    # ====================== NO-RELATION negatives (label 0) =================
    # curriculum: topic pairs that are INCOMPARABLE in the DAG (no path either way)
    incomparable = []
    node_list = sorted(n for n in nodes if n in by_sub)
    for i in range(len(node_list)):
        for j in range(i+1, len(node_list)):
            x, y = node_list[i], node_list[j]
            if y not in closure.get(x, ()) and x not in closure.get(y, ()):
                incomparable.append((x, y))
    rng.shuffle(incomparable)
    target_norel = int((wf_pos + cur_pos) * args.norel_ratio)
    made = 0
    bi = 0
    while made < target_norel and incomparable:
        x, y = incomparable[bi % len(incomparable)]; bi += 1
        if not by_sub[x] or not by_sub[y]:
            continue
        if sub_split.get(x) != sub_split.get(y):
            continue
        sp = sub_split.get(x, "train")
        ua, ta, txa = rng.choice(by_sub[x])
        ub, tb, txb = rng.choice(by_sub[y])
        emit(sp, ua, x, ta, txa, ub, y, tb, txb, 0, "norel")
        made += 1
        if bi > len(incomparable) * 50:    # safety
            break
    print(f"[norel] {made} no-relation negatives", file=sys.stderr)

    # ====================== write =========================================
    for sp in ("train", "val", "test"):
        rng.shuffle(rows[sp])
        with open(out / f"directional_{sp}.jsonl", "w") as f:
            for r in rows[sp]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # report
    def bal(sp):
        c = defaultdict(int); sc = defaultdict(int)
        for r in rows[sp]:
            c[r["label"]] += 1; sc[r["scale"]] += 1
        return dict(c), dict(sc)
    for sp in ("train", "val", "test"):
        c, sc = bal(sp)
        print(f"[done] {sp}: {len(rows[sp])} rows  labels={c}  scales={sc}",
              file=sys.stderr)

if __name__ == "__main__":
    main()

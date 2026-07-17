"""Merge directional_procedural + aep_gapdecay + ajo_gapdecay -> merged_gapdecay_procedural.

Three-way rebuild of the ~39k procedural+AEP+AJO training set, using the freshly
gap-decayed AEP (~13k) and AJO (~6.3k) tier-directional sets instead of the legacy
`ajo_newstyle` artifact. Adapted from scripts/merge_newstyle_procedural.py (paths are
now repo-relative, not hard-coded to /mnt/localssd/automation/...).

Output schema follows directional_procedural:
    text_1, text_2, label, step_1, step_2, url, title, source   (+ url_2 provenance)

- procedural rows already carry step_1/step_2/url/title (single-doc workflow).
- gap-decay rows carry sub_1/sub_2 + url_1/url_2 (cross-doc tier pair); we map
  sub_1/sub_2 -> step_1/step_2, url_1 -> url, keep url_2, and backfill title from
  the corpus (by url_1), falling back to any title seen in procedural.

Each split = concat of the three sources' matching split. Then:
  1. drop exact-duplicate (text_1, text_2, label) within a split,
  2. drop cross-split leaks on (text_1, text_2), keeping the earliest split
     (train > val > test).
"""
import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]          # internship-causal-embedding/
DATA_DIR = REPO / "data"
PROC_DIR = DATA_DIR / "new_aep_workflow_scrap" / "directional_procedural"
AEP_DIR = DATA_DIR / "aep_gapdecay"
AJO_DIR = DATA_DIR / "ajo_gapdecay"
OUT_DIR = DATA_DIR / "merged_gapdecay_procedural"
CORPUS = REPO / "src" / "followup_data" / "builders" / "aep_docs_collection_v_24-03-2026.json"

SPLITS = ("train", "val", "test")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def corpus_titles(path: Path) -> dict:
    """url -> title, via the builder's load_corpus (dependency-free)."""
    spec = importlib.util.spec_from_file_location(
        "_bcd", REPO / "src" / "followup_data" / "builders" / "build_classification_data.py")
    bcd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bcd)
    return {u: d.get("title", "") for u, d in bcd.load_corpus(path).items()}


def normalize_procedural(rows, source):
    out = []
    for r in rows:
        out.append({
            "text_1": r["text_1"], "text_2": r["text_2"], "label": r["label"],
            "step_1": r.get("step_1"), "step_2": r.get("step_2"),
            "url": r.get("url"), "url_2": r.get("url"),
            "title": r.get("title"), "source": source,
        })
    return out


def normalize_gapdecay(rows, source, url_title):
    out = []
    for r in rows:
        url1 = r.get("url_1")
        out.append({
            "text_1": r["text_1"], "text_2": r["text_2"], "label": r["label"],
            "step_1": r.get("sub_1"), "step_2": r.get("sub_2"),
            "url": url1, "url_2": r.get("url_2"),
            "title": url_title.get(url1), "source": source,
        })
    return out


def main():
    proc = {s: load_jsonl(PROC_DIR / f"directional_{s}.jsonl") for s in SPLITS}
    aep = {s: load_jsonl(AEP_DIR / f"directional_{s}.jsonl") for s in SPLITS}
    ajo = {s: load_jsonl(AJO_DIR / f"directional_{s}.jsonl") for s in SPLITS}

    print(f"[corpus] loading titles from {CORPUS.name}")
    url_title = corpus_titles(CORPUS)
    # also let procedural titles fill any gaps
    for rows in proc.values():
        for r in rows:
            if r.get("url") and r.get("title"):
                url_title.setdefault(r["url"], r["title"])

    merged = {}
    for s in SPLITS:
        merged[s] = (
            normalize_procedural(proc[s], "procedural")
            + normalize_gapdecay(aep[s], "aep", url_title)
            + normalize_gapdecay(ajo[s], "ajo", url_title)
        )

    # 1. exact-duplicate rows within a split
    for s in SPLITS:
        seen, deduped = set(), []
        for r in merged[s]:
            key = (r["text_1"], r["text_2"], r["label"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(r)
        dropped = len(merged[s]) - len(deduped)
        if dropped:
            print(f"{s}: dropped {dropped} exact-duplicate rows within split")
        merged[s] = deduped

    # 2. cross-split leaks on (text_1, text_2): keep earliest (train > val > test)
    seen_pairs = set()
    for s in SPLITS:
        kept, dropped = [], 0
        for r in merged[s]:
            key = (r["text_1"], r["text_2"])
            if key in seen_pairs:
                dropped += 1
                continue
            seen_pairs.add(key)
            kept.append(r)
        if dropped:
            print(f"{s}: dropped {dropped} rows leaking from an earlier split")
        merged[s] = kept

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        rows = merged[s]
        out_path = OUT_DIR / f"{s}.jsonl"
        with open(out_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        by_src = {src: sum(1 for r in rows if r["source"] == src)
                  for src in ("procedural", "aep", "ajo")}
        n_pos = sum(1 for r in rows if r["label"] == 1)
        print(f"{s}: {len(rows)} rows  {by_src}  "
              f"(label1={n_pos}, label0={len(rows) - n_pos}) -> {out_path}")


if __name__ == "__main__":
    main()

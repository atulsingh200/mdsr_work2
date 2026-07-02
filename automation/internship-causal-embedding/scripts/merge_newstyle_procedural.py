"""Merge directional_procedural and ajo_newstyle datasets into merged_newstyle_procedural.

Output schema follows directional_procedural: text_1, text_2, label, step_1, step_2, url, title.
ajo_newstyle rows don't have step_1/step_2/title, so heading_1/heading_2 fill in for
step_1/step_2, and title is borrowed from directional_procedural when the same url
appears there.

Each split is formed by simply concatenating the two datasets' matching split
(train+train, val+val, test+test). Leak check: after merging, if the exact same
(text_1, text_2) pair shows up in more than one split, it's dropped from the
later split (val/test dropped in favor of train, test dropped in favor of val)
so no exact-duplicate example leaks across splits.
"""
import json
from pathlib import Path

DATA_DIR = Path("/mnt/localssd/automation/internship-causal-embedding/data")
DIRECTIONAL_DIR = DATA_DIR / "new_aep_workflow_scrap" / "directional_procedural"
AJO_DIR = DATA_DIR / "ajo_newstyle"
OUT_DIR = DATA_DIR / "merged_newstyle_procedural"

SPLITS = ("train", "val", "test")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def normalize_directional(rows, source):
    out = []
    for r in rows:
        out.append({
            "text_1": r["text_1"],
            "text_2": r["text_2"],
            "label": r["label"],
            "step_1": r["step_1"],
            "step_2": r["step_2"],
            "url": r["url"],
            "title": r["title"],
            "source": source,
        })
    return out


def normalize_ajo(rows, source, url_title):
    out = []
    for r in rows:
        out.append({
            "text_1": r["text_1"],
            "text_2": r["text_2"],
            "label": r["label"],
            "step_1": r["heading_1"],
            "step_2": r["heading_2"],
            "url": r["url"],
            "title": url_title.get(r["url"]),
            "source": source,
        })
    return out


def main():
    directional = {s: load_jsonl(DIRECTIONAL_DIR / f"directional_{s}.jsonl") for s in SPLITS}
    ajo = {s: load_jsonl(AJO_DIR / f"{s}.jsonl") for s in SPLITS}

    url_title = {}
    for rows in directional.values():
        for r in rows:
            url_title.setdefault(r["url"], r["title"])

    merged = {}
    for s in SPLITS:
        merged[s] = normalize_directional(directional[s], "directional_procedural") + \
            normalize_ajo(ajo[s], "ajo_newstyle", url_title)

    # dedupe exact-duplicate rows within each split
    for s in SPLITS:
        seen = set()
        deduped = []
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

    # leak check across splits: same (text_1, text_2) pair in more than one split.
    # keep it in the earliest split (train > val > test), drop from the later one(s).
    seen_pairs = set()
    for s in SPLITS:
        kept = []
        dropped = 0
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
                f.write(json.dumps(r) + "\n")
        n_dir = sum(1 for r in rows if r["source"] == "directional_procedural")
        n_ajo = len(rows) - n_dir
        n_pos = sum(1 for r in rows if r["label"] == 1)
        print(f"{s}: {len(rows)} rows (directional={n_dir}, ajo={n_ajo}, label=1: {n_pos}, label=0: {len(rows) - n_pos}) -> {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Remove stop words from aep_causal_classification_34 dataset and write to a new directory.

Uses the full Onix Stop Word List 1 (429 words) plus custom AEP/Adobe domain stop words.

Usage:
  python3 remove_stopwords.py \
      --in-dir  data/aep_causal_classification_34 \
      --out-dir data/aep_causal_classification_34_nostop_full \
      [--lowercase]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Onix Stop Word List 1 — 429 words
# ---------------------------------------------------------------------------
ONIX_STOP_WORDS: set[str] = {
    "a", "about", "above", "across", "after", "again", "against", "all",
    "almost", "alone", "along", "already", "also", "although", "always",
    "among", "an", "and", "another", "any", "anybody", "anyone", "anything",
    "anywhere", "are", "area", "areas", "around", "as", "ask", "asked",
    "asking", "asks", "at", "away",
    "b", "back", "backed", "backing", "backs", "be", "became", "because",
    "become", "becomes", "been", "before", "began", "behind", "being",
    "beings", "best", "better", "between", "big", "both", "but", "by",
    "c", "came", "can", "cannot", "case", "cases", "certain", "certainly",
    "clear", "clearly", "come", "could",
    "d", "did", "differ", "different", "differently", "do", "does", "done",
    "down", "downed", "downing", "downs", "during",
    "e", "each", "early", "either", "end", "ended", "ending", "ends",
    "enough", "even", "evenly", "ever", "every", "everybody", "everyone",
    "everything", "everywhere",
    "f", "face", "faces", "fact", "facts", "far", "felt", "few", "find",
    "finds", "first", "for", "four", "from", "full", "fully", "further",
    "furthered", "furthering", "furthers",
    "g", "gave", "general", "generally", "get", "gets", "give", "given",
    "gives", "go", "going", "good", "goods", "got", "great", "greater",
    "greatest", "group", "grouped", "grouping", "groups",
    "h", "had", "has", "have", "having", "he", "her", "here", "herself",
    "high", "higher", "highest", "him", "himself", "his", "how", "however",
    "i", "if", "important", "in", "interest", "interested", "interesting",
    "interests", "into", "is", "it", "its", "itself",
    "j", "just",
    "k", "keep", "keeps", "kind", "knew", "know", "known", "knows",
    "l", "large", "largely", "last", "later", "latest", "least", "less",
    "let", "lets", "like", "likely", "long", "longer", "longest",
    "m", "made", "make", "making", "man", "many", "may", "me", "member",
    "members", "men", "might", "more", "most", "mostly", "mr", "mrs",
    "much", "must", "my", "myself",
    "n", "necessary", "need", "needed", "needing", "needs", "never", "new",
    "newer", "newest", "next", "no", "nobody", "non", "noone", "not",
    "nothing", "now", "nowhere", "number", "numbers",
    "o", "of", "off", "often", "old", "older", "oldest", "on", "once",
    "one", "only", "open", "opened", "opening", "opens", "or", "order",
    "ordered", "ordering", "orders", "other", "others", "our", "out",
    "over",
    "p", "part", "parted", "parting", "parts", "per", "perhaps", "place",
    "places", "point", "pointed", "pointing", "points", "possible",
    "present", "presented", "presenting", "presents", "problem", "problems",
    "put", "puts",
    "q", "quite",
    "r", "rather", "really", "right", "room", "rooms",
    "s", "said", "same", "saw", "say", "says", "second", "seconds", "see",
    "seem", "seemed", "seeming", "seems", "sees", "several", "shall", "she",
    "should", "show", "showed", "showing", "shows", "side", "sides",
    "since", "small", "smaller", "smallest", "so", "some", "somebody",
    "someone", "something", "somewhere", "state", "states", "still",
    "such", "sure",
    "t", "take", "taken", "than", "that", "the", "their", "them", "then",
    "there", "therefore", "these", "they", "thing", "things", "think",
    "thinks", "this", "those", "though", "thought", "thoughts", "three",
    "through", "thus", "to", "today", "together", "too", "took", "toward",
    "turn", "turned", "turning", "turns", "two",
    "u", "under", "until", "up", "upon", "us", "use", "used", "uses",
    "v", "very",
    "w", "want", "wanted", "wanting", "wants", "was", "way", "ways", "we",
    "well", "wells", "went", "were", "what", "when", "where", "whether",
    "which", "while", "who", "whole", "whose", "why", "will", "with",
    "within", "without", "work", "worked", "working", "works", "would",
    "x",
    "y", "year", "years", "yet", "you", "young", "younger", "youngest",
    "your", "yours",
    "z",
}

# ---------------------------------------------------------------------------
# Custom AEP / Adobe domain stop words
# These are high-frequency but low-information words in the AEP/AJO corpus:
# UI chrome, documentation boilerplate, product name repetition, and
# transcript filler that carry no causal-tier signal.
# ---------------------------------------------------------------------------
CUSTOM_STOP_WORDS: set[str] = {
    # Product / brand names that appear on nearly every page
    "adobe", "aep", "ajo", "journey", "optimizer", "experience", "platform",
    "experienceleague", "experienceplatform",

    # UI chrome / navigation boilerplate
    "click", "select", "navigate", "go", "open", "close", "menu", "tab",
    "button", "panel", "page", "screen", "window", "dialog", "modal",
    "sidebar", "dropdown", "section", "toggle", "checkbox", "icon",
    "scroll", "expand", "collapse", "view", "overview",

    # Documentation / tutorial boilerplate
    "learn", "tutorial", "guide", "documentation", "doc", "docs", "video",
    "transcript", "following", "following", "step", "steps", "follow",
    "note", "tip", "warning", "example", "demo", "sample", "see",
    "refer", "reference", "information", "details", "info", "help",
    "https", "http", "www", "com", "html",
    "recommendation", "more",  # from "recommendation-more-help" artifacts

    # Generic action verbs with no domain meaning in isolation
    "create", "add", "edit", "update", "delete", "remove", "save",
    "configure", "set", "define", "specify", "enter", "type", "input",
    "confirm", "submit", "apply", "enable", "disable", "allow", "prevent",
    "access", "manage", "review", "check", "verify", "test", "run",
    "start", "stop", "launch", "deploy", "publish", "preview",
    "upload", "download", "import", "export", "send", "receive", "copy",

    # Ubiquitous AJO nouns that appear in every tier
    "data", "user", "users", "customer", "customers", "message", "messages",
    "email", "emails", "channel", "channels", "campaign", "campaigns",
    "profile", "profiles", "event", "events", "action", "actions",
    "content", "feature", "features", "service", "services",
    "option", "options", "value", "values", "field", "fields",
    "name", "type", "id", "list", "item", "items", "object", "objects",
    "property", "properties", "attribute", "attributes",
    "result", "results", "report", "reports", "log", "logs",
    "flow", "process", "workflow", "task", "tasks",
    "time", "date", "day", "month", "week",
    "new", "available", "current", "default", "custom",
    "based", "using", "used", "allows", "provides", "includes",
    "within", "across", "multiple", "single", "specific", "additional",
    "need", "needs", "want", "wants", "able", "click",

    # Numeric / symbol artifacts from chunking
    "recommendation-more-help",
}

# Combined stop word set (all lowercase)
ALL_STOP_WORDS: set[str] = ONIX_STOP_WORDS | CUSTOM_STOP_WORDS

# Tokenise on whitespace + strip punctuation from token boundaries
_TOKEN_RE = re.compile(r"\S+")
_PUNCT_STRIP = re.compile(r"^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$")


def remove_stopwords(text: str) -> str:
    """Remove stop words from *text*, preserving original token casing."""
    tokens = _TOKEN_RE.findall(text)
    kept: list[str] = []
    for tok in tokens:
        # Strip leading/trailing punctuation just for the lookup
        core = _PUNCT_STRIP.sub("", tok).lower()
        if core not in ALL_STOP_WORDS:
            kept.append(tok)
    return " ".join(kept)


def process_split(in_path: Path, out_path: Path, fields: tuple[str, ...]) -> int:
    """Process one JSONL split file. Returns number of records written."""
    count = 0
    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for field in fields:
                if field in record and isinstance(record[field], str):
                    record[field] = remove_stopwords(record[field])
            fout.write(json.dumps(record) + "\n")
            count += 1
    return count


def main() -> None:
    ap = argparse.ArgumentParser(description="Strip stop words from classification JSONL splits.")
    ap.add_argument("--in-dir",  default="data/aep_causal_classification_34",
                    help="Source dataset directory (default: data/aep_causal_classification_34)")
    ap.add_argument("--out-dir", default="data/aep_causal_classification_34_nostop_full",
                    help="Output directory (default: data/aep_causal_classification_34_nostop_full)")
    ap.add_argument("--fields", default="text_1,text_2",
                    help="Comma-separated JSONL fields to strip (default: text_1,text_2)")
    ap.add_argument("--splits", default="train,val,test",
                    help="Comma-separated split names (default: train,val,test)")
    ap.add_argument("--list-stopwords", action="store_true",
                    help="Print the full combined stop word list and exit.")
    args = ap.parse_args()

    if args.list_stopwords:
        for w in sorted(ALL_STOP_WORDS):
            print(w)
        print(f"\nTotal: {len(ALL_STOP_WORDS)} stop words "
              f"({len(ONIX_STOP_WORDS)} Onix + "
              f"{len(CUSTOM_STOP_WORDS)} custom, "
              f"{len(ONIX_STOP_WORDS & CUSTOM_STOP_WORDS)} overlap)")
        return

    in_dir  = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    fields  = tuple(args.fields.split(","))
    splits  = args.splits.split(",")

    if not in_dir.exists():
        print(f"[error] input directory not found: {in_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[stopwords] {len(ALL_STOP_WORDS)} total "
          f"({len(ONIX_STOP_WORDS)} Onix + {len(CUSTOM_STOP_WORDS)} custom, "
          f"{len(ONIX_STOP_WORDS & CUSTOM_STOP_WORDS)} overlap)", file=sys.stderr)
    print(f"[fields]    stripping: {fields}", file=sys.stderr)
    print(f"[in-dir]    {in_dir}", file=sys.stderr)
    print(f"[out-dir]   {out_dir}", file=sys.stderr)

    for split in splits:
        fname = f"directional_{split}.jsonl"
        src = in_dir / fname
        dst = out_dir / fname
        if not src.exists():
            print(f"[skip] {src} not found", file=sys.stderr)
            continue
        n = process_split(src, dst, fields)
        print(f"[done] {split:5s}  {n:>7,d} records  -> {dst}", file=sys.stderr)

    # Copy manifest as-is so downstream tools can inspect the source config
    manifest_src = in_dir / "directional_manifest.json"
    if manifest_src.exists():
        manifest_dst = out_dir / "directional_manifest.json"
        shutil.copy2(manifest_src, manifest_dst)
        # Append a note about stop word removal
        manifest = json.loads(manifest_dst.read_text())
        manifest["stopword_removal"] = {
            "applied": True,
            "fields": list(fields),
            "n_onix": len(ONIX_STOP_WORDS),
            "n_custom": len(CUSTOM_STOP_WORDS),
            "n_total": len(ALL_STOP_WORDS),
            "source_dir": str(in_dir),
        }
        manifest_dst.write_text(json.dumps(manifest, indent=2))
        print(f"[done] manifest written to {manifest_dst}", file=sys.stderr)

    print("[done] all splits processed.", file=sys.stderr)


if __name__ == "__main__":
    main()

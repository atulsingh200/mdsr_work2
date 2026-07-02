#!/usr/bin/env bash
# Build the survey PDF. Merges per-section bib files into refs.bib, then runs tectonic.
set -e
cd "$(dirname "$0")"

echo ">> merging bib fragments into refs.bib"
cat bib/*.bib > refs.bib 2>/dev/null || true
echo "   $(grep -c '^@' refs.bib 2>/dev/null || echo 0) bib entries"

# Stub any missing section files so the doc always compiles.
for f in 00_abstract 01_introduction 02_foundations 03_embeddings_retrieval \
         04_causal_reasoning 05_asymmetric_geometry 06_reasoning_rerankers_rl \
         06b_latent_reasoning 06c_reasoning_embeddings \
         07_ordering_aggregation 08_followup_conversational 09_procedural_workflow \
         10_data_characterization 11_system_design 12_conclusion; do
  if [ ! -f "sections/$f.tex" ]; then
    echo "%% placeholder for $f" > "sections/$f.tex"
  fi
done

echo ">> compiling with tectonic (handles bibtex passes automatically)"
TEC=./tectonic
[ -x "$TEC" ] || TEC=tectonic
"$TEC" -X compile main.tex --keep-logs 2>&1 | tail -25

echo ">> done: $(ls -lh main.pdf 2>/dev/null | awk '{print $5}')"

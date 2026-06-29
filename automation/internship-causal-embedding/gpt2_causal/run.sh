#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="${1:-config.yaml}"
PYTHON="$(which python3)"

echo "============================================"
echo " GPT-2 Causal Classifier"
echo " Config : $CONFIG"
echo " Python : $PYTHON"
echo " Date   : $(date)"
echo "============================================"

# ── Smoke test ──────────────────────────────────
echo ""
echo "[1/3] Smoke test (10 steps, 100 examples)..."
$PYTHON train.py --config "$CONFIG" --smoke-test
echo "Smoke test passed."

# ── Full training ────────────────────────────────
echo ""
echo "[2/3] Full training..."
$PYTHON train.py --config "$CONFIG"

# ── Evaluation ───────────────────────────────────
CHECKPOINT_DIR="$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['output_dir'])")/best_model"
VAL_PATH="$(python3  -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['val_path'])")"
TEST_PATH="$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['test_path'])")"

echo ""
echo "[3/3] Evaluating best checkpoint: $CHECKPOINT_DIR"

echo ""
echo "--- Validation set ---"
$PYTHON evaluate.py --checkpoint "$CHECKPOINT_DIR" --data "$VAL_PATH"  --config "$CONFIG"

echo ""
echo "--- Test set ---"
$PYTHON evaluate.py --checkpoint "$CHECKPOINT_DIR" --data "$TEST_PATH" --config "$CONFIG"

echo ""
echo "============================================"
echo " Done. $(date)"
echo "============================================"

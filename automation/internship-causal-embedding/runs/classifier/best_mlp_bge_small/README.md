
# Classifier Run: best_mlp_bge_small

Directional causal classifier trained on AEP/AJO tutorial segments.  
Task: given two text segments, predict whether `text_1`'s tier causally precedes `text_2`'s tier in a 15-tier dependency hierarchy.

---

## Overall Test Results

| Metric   | Score  |
|----------|--------|
| Accuracy | 0.8571 |
| AUC      | 0.9031 |
| F1       | 0.8558 |
| Loss     | 1.1182 |
| n (test) | 6,004  |

---

## Model & Training Configuration

| Parameter        | Value                              |
|------------------|------------------------------------|
| Base encoder     | `BAAI/bge-small-en-v1.5` (384-dim) |
| Head             | MLP: 4d → 512 → 128 → 1           |
| Activation       | GELU                               |
| Norm             | LayerNorm (pre-norm per layer)     |
| Dropout          | 0.1                                |
| Pooling          | CLS token                          |
| L2 normalize     | Yes                                |
| Feature concat   | [A; B; A−B; A⊙B]                  |
| Epochs           | 3                                  |
| Batch size       | 32                                 |
| LR (encoder)     | 2e-5                               |
| LR (head)        | 1e-3                               |
| Warmup fraction  | 10%                                |
| Weight decay     | 0.01                               |
| Grad clip        | 1.0                                |
| Max seq len      | 524                                |
| Precision        | bfloat16 (CUDA)                    |
| Seed             | 42                                 |
| Freeze encoders  | No (full fine-tune)                |
| Tier aux loss    | Disabled                           |

**Dataset splits:**

| Split | Rows    |
|-------|---------|
| Train | 110,775 |
| Val   |   6,004 |
| Test  |   6,004 |

Labels are pre-assigned: `label=1` means `text_1` is from a lower-numbered (prerequisite) tier; `label=0` is the reversed direction. Negatives are structural — each chunk pair appears in exactly one direction (`--single-direction`), yielding a ~50/50 class balance.  
Loss: `BCEWithLogitsLoss`.

---

## Training Progression

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val AUC |
|-------|-----------|-----------|----------|---------|---------|
| 1     | 0.1671    | 0.9271    | 0.7209   | 0.8718  | 0.9271  |
| 2     | 0.0291    | 0.9876    | 0.8319   | 0.8841  | 0.9258  |
| 3     | 0.0172    | 0.9907    | 0.8747   | **0.8904** | **0.9318** |

Best checkpoint saved at **epoch 3** (`val_acc = 0.8904`). Test evaluation uses `best.pt`.

---

## Tier Reference

| Tier | Name                                   |
|------|----------------------------------------|
| T1   | Product orientation                    |
| T2   | Admin floor                            |
| T3   | Data foundations                       |
| T4   | Profiles, audiences & subscriptions   |
| T5   | Content authoring foundations          |
| T6   | Channels + campaigns (first sends)     |
| T7   | Journeys (core)                        |
| T8   | Personalization & decisioning          |
| T9   | Advanced journey patterns & experimentation |
| T10  | Multi-journey orchestration            |
| T11  | Admin configuration                    |
| T12  | Governance & privacy                   |
| T13  | Observability & reporting              |
| T14  | AI agents & assistants                 |
| T15  | Use cases, labs, capstones             |

---

## Per-Tier-Pair Test Accuracy

All 105 tier-pair combinations (C(15,2)). `n` = number of test samples for that pair.

| Pair    |   n | Acc    | Pair    |   n | Acc    | Pair    |   n | Acc    |
|---------|-----|--------|---------|-----|--------|---------|-----|--------|
| T1-T2   |   2 | 0.0000 | T4-T5   |  40 | 0.3500 | T7-T8   |  54 | 0.6111 |
| T1-T3   |  10 | 0.5000 | T4-T6   | 128 | 0.4688 | T7-T9   |  18 | 0.6111 |
| T1-T4   |   8 | 0.7500 | T4-T7   |  12 | 0.8333 | T7-T10  |   6 | 0.5000 |
| T1-T5   |  20 | 0.5000 | T4-T8   |  72 | 0.6806 | T7-T11  |  36 | 0.6111 |
| T1-T6   |  64 | 0.6875 | T4-T9   |  24 | 0.7083 | T7-T12  |  12 | 0.6667 |
| T1-T7   |   6 | 1.0000 | T4-T10  |   8 | 0.6250 | T7-T13  |  12 | 0.6667 |
| T1-T8   |  36 | 0.9167 | T4-T11  |  48 | 0.7500 | T7-T14  |   3 | 0.3333 |
| T1-T9   |  12 | 1.0000 | T4-T12  |  16 | 1.0000 | T7-T15  |  42 | 0.7619 |
| T1-T10  |   4 | 1.0000 | T4-T13  |  16 | 0.9375 | T8-T9   | 108 | 0.8148 |
| T1-T11  |  24 | 0.9583 | T4-T14  |   4 | 0.5000 | T8-T10  |  36 | 0.7500 |
| T1-T12  |   8 | 1.0000 | T4-T15  |  56 | 0.8571 | T8-T11  | 216 | 0.8426 |
| T1-T13  |   8 | 1.0000 | T5-T6   | 320 | 0.7875 | T8-T12  |  72 | 0.9861 |
| T1-T14  |   2 | 1.0000 | T5-T7   |  30 | 0.9667 | T8-T13  |  72 | 0.9861 |
| T1-T15  |  28 | 0.9643 | T5-T8   | 180 | 0.9944 | T8-T14  |  18 | 0.0556 |
| T2-T3   |   5 | 1.0000 | T5-T9   |  60 | 1.0000 | T8-T15  | 252 | 0.8254 |
| T2-T4   |   4 | 1.0000 | T5-T10  |  20 | 1.0000 | T9-T10  |  12 | 0.7500 |
| T2-T5   |  10 | 1.0000 | T5-T11  | 120 | 0.9500 | T9-T11  |  72 | 0.7361 |
| T2-T6   |  32 | 1.0000 | T5-T12  |  40 | 1.0000 | T9-T12  |  24 | 0.9583 |
| T2-T7   |   3 | 1.0000 | T5-T13  |  40 | 1.0000 | T9-T13  |  24 | 1.0000 |
| T2-T8   |  18 | 1.0000 | T5-T14  |  10 | 1.0000 | T9-T14  |   6 | 0.0000 |
| T2-T9   |   6 | 1.0000 | T5-T15  | 140 | 0.9357 | T9-T15  |  84 | 0.8452 |
| T2-T10  |   2 | 1.0000 | T6-T7   |  96 | 0.9479 | T10-T11 |  24 | 0.7917 |
| T2-T11  |  12 | 1.0000 | T6-T8   | 576 | 0.9306 | T10-T12 |   8 | 1.0000 |
| T2-T12  |   4 | 1.0000 | T6-T9   | 192 | 0.9479 | T10-T13 |   8 | 1.0000 |
| T2-T13  |   4 | 1.0000 | T6-T10  |  64 | 0.8750 | T10-T14 |   2 | 0.0000 |
| T2-T14  |   1 | 1.0000 | T6-T11  | 384 | 0.8828 | T10-T15 |  28 | 0.9286 |
| T2-T15  |  14 | 1.0000 | T6-T12  | 128 | 0.9922 | T11-T12 |  48 | 0.8333 |
| T3-T4   |  20 | 0.8500 | T6-T13  | 128 | 1.0000 | T11-T13 |  48 | 1.0000 |
| T3-T5   |  50 | 0.7000 | T6-T14  |  32 | 0.9063 | T11-T14 |  12 | 0.0833 |
| T3-T6   | 160 | 0.7500 | T6-T15  | 448 | 0.9174 | T11-T15 | 168 | 0.8631 |
| T3-T7   |  15 | 0.8000 | T7-T8   |  54 | 0.6111 | T12-T13 |  16 | 1.0000 |
| T3-T8   |  90 | 0.7444 | T7-T9   |  18 | 0.6111 | T12-T14 |   4 | 0.0000 |
| T3-T9   |  30 | 0.6667 | T7-T10  |   6 | 0.5000 | T12-T15 |  56 | 0.7857 |
| T3-T10  |  10 | 0.6000 | T7-T11  |  36 | 0.6111 | T13-T14 |   4 | 0.0000 |
| T3-T11  |  60 | 0.8833 | T7-T12  |  12 | 0.6667 | T13-T15 |  56 | 0.7321 |
| T3-T12  |  20 | 1.0000 | T7-T13  |  12 | 0.6667 | T14-T15 |  14 | 0.9286 |
| T3-T13  |  20 | 1.0000 | T7-T14  |   3 | 0.3333 |         |     |        |
| T3-T14  |   5 | 0.8000 | T7-T15  |  42 | 0.7619 |         |     |        |
| T3-T15  |  70 | 0.9714 |         |     |        |         |     |        |

---

## Analysis

**Strong pairs (acc ≥ 0.95)** — large tier distance or semantically distinct domains make direction easy to infer:
- All T2-Tx pairs (Admin floor vs. everything): 100%
- T5-T8, T5-T9, T5-T10, T5-T12, T5-T13, T5-T14: 99–100%
- T6-T13 (Channels vs. Reporting): 100%
- T8-T12, T8-T13 (Decisioning vs. Governance/Reporting): 98.6%

**Weak pairs (acc < 0.55)** — semantically adjacent tiers or low sample counts:
- **T4-T5** (Profiles vs. Content authoring): 35.0% — these tiers are practically co-dependent in AJO workflows
- **T4-T6** (Profiles vs. Channels): 46.9% — similar issue; both are mid-funnel setup steps
- **T8-T14** (Decisioning vs. AI agents): 5.6% — T14 content heavily overlaps with T8 language
- **T9-T14** (Advanced journeys vs. AI agents): 0.0% — only 6 samples, likely noise
- **T10-T14**, **T12-T14**, **T13-T14**: 0.0% — T14 (AI Agents) is consistently confused with adjacent tiers; its tutorial language borrows from all surrounding tiers
- **T11-T14** (Admin config vs. AI agents): 8.3%
- **T1-T2** (Product orientation vs. Admin): 0.0% — only 2 samples

**T7 (Journeys core) is a consistent weak spot** across all its pairs (0.33–0.76 range), likely because journey-building language appears throughout all tier levels.

---

## Output Files

| File                     | Description                                              |
|--------------------------|----------------------------------------------------------|
| `config.json`            | Full hyperparameter and architecture specification       |
| `train.log`              | Per-step loss and per-epoch val metrics                  |
| `test_metrics.json`      | Overall test scores + per-tier-pair accuracy breakdown   |
| `test_predictions.jsonl` | One row per test example with `prob`, `label`, tier info |
| `best.pt`                | Best checkpoint (epoch 3, val_acc=0.8904) — not tracked  |
| `final.pt`               | Final epoch checkpoint — not tracked                     |

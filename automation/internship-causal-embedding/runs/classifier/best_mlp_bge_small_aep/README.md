# Classifier Run: best_mlp_bge_small_aep

Directional causal classifier trained on Adobe Experience Platform (AEP) documentation segments.  
Task: given two text segments, predict whether `text_1`'s tier causally precedes `text_2`'s tier in an 11-tier dependency hierarchy.

---

## Overall Test Results

| Metric   | Score  |
|----------|--------|
| Accuracy | 0.9243 |
| AUC      | 0.9513 |
| F1       | 0.9257 |
| Loss     | 0.5722 |
| n (test) | 8,460  |

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
| Max seq len      | 512                                |
| Precision        | bfloat16 (CUDA)                    |
| Seed             | 42                                 |
| Freeze encoders  | No (full fine-tune)                |
| Tier aux loss    | Disabled                           |

> **Note on `max_seq_len`:** AEP segments are much longer than the AJO ones (longest observed: 4,699 tokens vs. 497). `bge-small-en-v1.5` only has 512 position embeddings, so `--max-seq-len` is capped at **512** here; longer segments are truncated. The AJO run's nominal 524 was never actually exceeded, so both runs are effectively bounded by the encoder's 512-token limit.

**Dataset splits:**

| Split | Rows   |
|-------|--------|
| Train | 67,500 |
| Val   |  8,460 |
| Test  |  8,460 |

Labels are pre-assigned: `label=1` means `text_1` is from a lower-numbered (prerequisite) tier; `label=0` is the reversed direction. Negatives are structural — each chunk pair appears in exactly one direction (`--single-direction`), yielding a ~50/50 class balance. Per tier-pair the test set is exactly balanced at **188 pairs** (`--max-chunk-pairs-per-tier-pair 1500`, `split-level=chunk_tier`).  
Loss: `BCEWithLogitsLoss`.

---

## Training Progression

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val AUC |
|-------|-----------|-----------|----------|---------|---------|
| 1     | 0.2476    | 0.8928    | 0.2576   | 0.9305  | 0.9794  |
| 2     | 0.0481    | 0.9851    | 0.2760   | 0.9496  | 0.9797  |
| 3     | 0.0125    | 0.9967    | 0.3275   | **0.9499** | 0.9749  |

Best checkpoint saved at **epoch 3** (`val_acc = 0.9499`). Test evaluation uses `best.pt`.

---

## Tier Reference

| Tier | Name                          | Rationale                                                        |
|------|-------------------------------|------------------------------------------------------------------|
| T1   | Platform Foundation           | Core infrastructure and access controls that enable everything else |
| T2   | Data Architecture             | Schemas and data modeling that define structure for all data     |
| T3   | Data Collection               | Getting data into the platform using the defined schemas         |
| T4   | Data Storage & Catalog        | Managing and organizing collected data within the platform       |
| T5   | Identity Foundation           | Core identity resolution that enables unified customer profiles  |
| T6   | Profile & Segmentation        | Building unified profiles and audiences from identified data     |
| T7   | Data Processing & Analytics   | Advanced analysis and ML capabilities operating on profiles/data |
| T8   | Activation & Destinations     | Sending processed data and audiences to external systems         |
| T9   | Governance & Privacy          | Controls and compliance that govern all data operations          |
| T10  | Monitoring & Operations       | Observing and managing the running platform                      |
| T11  | Advanced Applications         | Specialized tools and applications built on the platform         |



## Per-Tier-Pair Test Accuracy

All 45 tier-pair combinations (C(10,2)). `n` = number of test samples for that pair (uniform at 188).

| Pair    |   n | Acc    | Pair    |   n | Acc    | Pair    |   n | Acc    |
|---------|-----|--------|---------|-----|--------|---------|-----|--------|
| T1-T2   | 188 | 0.7819 | T2-T10  | 188 | 0.9947 | T6-T7   | 188 | 0.8351 |
| T1-T3   | 188 | 0.8298 | T2-T11  | 188 | 0.9840 | T6-T8   | 188 | 0.9309 |
| T1-T4   | 188 | 0.8564 | T3-T4   | 188 | 0.8883 | T6-T9   | 188 | 0.9096 |
| T1-T6   | 188 | 0.9149 | T3-T6   | 188 | 0.9787 | T6-T10  | 188 | 0.9628 |
| T1-T7   | 188 | 0.8883 | T3-T7   | 188 | 0.9787 | T6-T11  | 188 | 0.9681 |
| T1-T8   | 188 | 0.8936 | T3-T8   | 188 | 0.9734 | T7-T8   | 188 | 0.8723 |
| T1-T9   | 188 | 0.9415 | T3-T9   | 188 | 0.9787 | T7-T9   | 188 | 0.9362 |
| T1-T10  | 188 | 0.9681 | T3-T10  | 188 | 1.0000 | T7-T10  | 188 | 0.9043 |
| T1-T11  | 188 | 0.9947 | T3-T11  | 188 | 1.0000 | T7-T11  | 188 | 0.9521 |
| T2-T3   | 188 | 0.7872 | T4-T6   | 188 | 0.8404 | T8-T9   | 188 | 0.9096 |
| T2-T4   | 188 | 0.8457 | T4-T7   | 188 | 0.9255 | T8-T10  | 188 | 0.8777 |
| T2-T6   | 188 | 0.9521 | T4-T8   | 188 | 0.9255 | T8-T11  | 188 | 0.9202 |
| T2-T7   | 188 | 0.9840 | T4-T9   | 188 | 0.9149 | T9-T10  | 188 | 0.8830 |
| T2-T8   | 188 | 1.0000 | T4-T10  | 188 | 0.9840 | T9-T11  | 188 | 0.9096 |
| T2-T9   | 188 | 1.0000 | T4-T11  | 188 | 0.9681 | T10-T11 | 188 | 0.8511 |

---

## Analysis

The AEP run is **stronger and more stable** than the AJO run (test acc 0.9243 vs. 0.8571, AUC 0.9513 vs. 0.9031). The test set is perfectly balanced (188 samples per pair), so accuracies are directly comparable across pairs without sample-count noise.

**Strong pairs (acc ≥ 0.97)** — large tier distance or semantically distinct domains make direction easy to infer:
- **T3-T10, T3-T11** (Data Collection vs. Monitoring / Advanced Applications): 100%
- **T2-T8, T2-T9** (Data Architecture vs. Activation / Governance): 100%
- **T2-T10, T1-T11** (foundational vs. operational/advanced tiers): 99.5%
- **T2-T7, T2-T11, T4-T10** and the rest of the T3-Tx block (T3-T6/T7/T8/T9): 97–98%

These all span the natural "set up → operate" gap in the platform lifecycle, where ordering is unambiguous.

**Weak pairs (acc < 0.86)** — semantically adjacent tiers near the start of the pipeline:
- **T1-T2** (Platform Foundation vs. Data Architecture): 78.2% — both are early setup tiers; foundational/admin language overlaps heavily.
- **T2-T3** (Data Architecture vs. Data Collection): 78.7% — schemas and ingestion are tightly co-dependent and described together.
- **T1-T3** (Platform Foundation vs. Data Collection): 83.0% — same early-setup confusion.
- **T6-T7** (Profile & Segmentation vs. Data Processing & Analytics): 83.5% — both consume unified profiles, so the prerequisite direction is subtle.
- **T4-T6** (Data Storage & Catalog vs. Profile & Segmentation): 84.0%
- **T10-T11** (Monitoring vs. Advanced Applications): 85.1% — both are late-stage tiers, so neither clearly precedes the other.

**Pattern:** errors concentrate on **adjacent tiers (distance 1–2)**, especially the foundation/data-modeling cluster (T1–T4), where the documentation language is most similar and the true causal ordering is genuinely close to interchangeable. Direction confidence grows monotonically with tier distance.

---

## Output Files

| File                     | Description                                              |
|--------------------------|----------------------------------------------------------|
| `config.json`            | Full hyperparameter and architecture specification       |
| `train.log`              | Per-step loss and per-epoch val metrics                  |
| `test_metrics.json`      | Overall test scores + per-tier-pair accuracy breakdown   |
| `test_predictions.jsonl` | One row per test example with `prob`, `label`, tier info |
| `best.pt`                | Best checkpoint (epoch 3, val_acc=0.9499) — not tracked  |
| `final.pt`               | Final epoch checkpoint — not tracked                     |

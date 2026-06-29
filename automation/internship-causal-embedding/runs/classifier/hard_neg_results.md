# Hard Negative Evaluation Results

Model: MLP classifier fine-tuned on BGE-small-en-v1.5 (two BERT towers).

## Accuracy Comparison

| Training Data | Test Data | AJO Acc | AEP Acc |
|---|---|---|---|
| Original dataset | Original test set | 0.8571 | 0.9243 |
| Original dataset | Hard negative test set | 0.7977 | 0.8520 |
| Hard negative dataset | Hard negative test set | 0.8260 | 0.8654 |

## Notes

- **Original dataset**: standard directional pairs where label=0 is a direction-flipped version of the positive
- **Hard negative dataset**: augmented with hard negatives — for each positive `x → y` (label=1), a hard negative `x → y*` (label=0) is added where `y*` is the most semantically similar chunk to `y` that has `tier(y*) < tier(x)` (wrong causal direction but similar content)
- **Hard negative test set**: same construction as above applied to the test split — measures how well the model rejects semantically similar but wrong-direction pairs
- Drop from original→hard-neg test (rows 1→2) shows the difficulty of the hard negatives: **-5.9 pp for AJO, -7.2 pp for AEP**
- Fine-tuning on hard negatives (row 3) recovers most of that gap: **+2.8 pp for AJO, +1.3 pp for AEP** over the baseline on the same hard-neg test set

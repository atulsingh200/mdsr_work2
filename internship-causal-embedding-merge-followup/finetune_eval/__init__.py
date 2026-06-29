"""Fine-tuning pipeline with semantic hard negatives.

Trains the two-tower bi-encoder from `src/biencoder/` on each `data_6/`
dataset that has a train split (aep_causal, followupqg, multiwoz_v24,
qrecc, workflow — aep_followup is excluded because it has no train split).

For each anchor x with true positive y, we mine K (=4) semantic-similar
*other* positives y* to use as hard negatives in the InfoNCE objective:

    x → y       label = 1   (true positive)
    x → y*_k    label = 0   (4 hard negatives, k = 1..4)

The miner uses a frozen sentence encoder (all-MiniLM-L6-v2) to find each
positive's top-K nearest neighbours in positive-space, then maps those
back to the corresponding (other anchor's) positive texts.

Pipeline:
    mine_hard_negatives.py   →   <dataset>/hard_negatives.npy
    train_finetune.py        →   <dataset>/checkpoint_best.pt
    evaluate_finetune.py     →   <dataset>/eval.json
"""

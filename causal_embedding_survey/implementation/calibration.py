"""calibration.py -- temperature scaling for pairwise causal scores (Idea 2).

Robust aggregation (ordering.py) consumes P[i,j] as a *probability/weight* (log-odds),
not just a sign. Over/under-confident scores distort weighted-FAS and Bradley-Terry
objectives and break any thresholding/abstention. Temperature scaling (Guo et al.,
ICML 2017) fixes this with a single scalar T>0: calibrated prob = sigmoid(logit / T).
Because it scales all logits equally it preserves the arg-max, so pairwise accuracy is
unchanged -- only confidences are corrected.

Functions
---------
- ``fit_temperature(logits, labels)``  : find T minimising validation NLL (scipy.optimize).
- ``apply_temperature(logits, T)``     : sigmoid(logits / T) -> calibrated probabilities.
- ``probs_to_logits`` / ``logits_to_probs`` : convenience conversions.
- ``expected_calibration_error(probs, labels)`` : binned ECE.
- ``reliability_diagram(probs, labels, path)``  : save a reliability-diagram PDF (matplotlib).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.optimize import minimize_scalar

_EPS = 1e-7


def logits_to_probs(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=float)))


def probs_to_logits(probs: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probs, dtype=float), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _nll(T: float, logits: np.ndarray, labels: np.ndarray) -> float:
    """Binary NLL of labels under sigmoid(logits / T)."""
    T = max(T, _EPS)
    p = np.clip(1.0 / (1.0 + np.exp(-logits / T)), _EPS, 1.0 - _EPS)
    return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))


def fit_temperature(logits: np.ndarray, labels: np.ndarray,
                    bounds: Tuple[float, float] = (0.05, 20.0)) -> float:
    """Fit a single temperature T>0 minimising validation NLL via scipy bounded search.

    ``logits`` are the raw (pre-sigmoid) pairwise scores; ``labels`` are 0/1 gold
    precedence labels for the same val pairs. Returns the optimal scalar T.
    """
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=float)
    res = minimize_scalar(_nll, args=(logits, labels), bounds=bounds, method="bounded")
    return float(res.x)


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Apply temperature scaling: calibrated probability = sigmoid(logits / T)."""
    T = max(float(T), _EPS)
    return 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=float) / T))


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Binned Expected Calibration Error (ECE)."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(probs)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs > lo) & (probs <= hi) if lo > 0 else (probs >= lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        conf = probs[mask].mean()
        acc = labels[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def reliability_diagram(probs: np.ndarray, labels: np.ndarray, path: str,
                        n_bins: int = 10, title: str = "Reliability diagram") -> str:
    """Plot and save a reliability diagram (bin accuracy vs confidence) as a PDF."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers, accs, confs = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs > lo) & (probs <= hi) if lo > 0 else (probs >= lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        centers.append((lo + hi) / 2)
        accs.append(labels[mask].mean())
        confs.append(probs[mask].mean())

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    ax.bar(centers, accs, width=1.0 / n_bins * 0.9, alpha=0.7,
           edgecolor="black", label="accuracy")
    ax.scatter(confs, accs, color="red", zorder=5, label="confidence vs acc")
    ece = expected_calibration_error(probs, labels, n_bins)
    ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
    ax.set_title(f"{title}  (ECE={ece:.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("calibration.py demo")
    rng = np.random.default_rng(0)
    n = 4000
    # true latent logits, but the model is OVERCONFIDENT (logits inflated by 3x)
    true_logit = rng.normal(0, 1.5, n)
    labels = (rng.uniform(size=n) < logits_to_probs(true_logit)).astype(float)
    model_logits = true_logit * 3.0  # overconfident

    raw_probs = logits_to_probs(model_logits)
    ece_before = expected_calibration_error(raw_probs, labels)
    T = fit_temperature(model_logits, labels)
    cal_probs = apply_temperature(model_logits, T)
    ece_after = expected_calibration_error(cal_probs, labels)

    print(f"fitted temperature T = {T:.3f}  (expect ~3.0 to undo 3x inflation)")
    print(f"ECE before = {ece_before:.4f}")
    print(f"ECE after  = {ece_after:.4f}")
    acc_before = ((raw_probs > 0.5) == labels).mean()
    acc_after = ((cal_probs > 0.5) == labels).mean()
    print(f"accuracy preserved: before {acc_before:.4f} == after {acc_after:.4f}")

    out = "/tmp/claude-1000/-mnt-localssd/2d16be33-ccdd-4ffd-b324-ab2606321924/scratchpad/reliability_demo.pdf"
    try:
        reliability_diagram(cal_probs, labels, out, title="Calibrated")
        print(f"saved reliability diagram -> {out}")
    except Exception as e:  # pragma: no cover
        print(f"(reliability diagram skipped: {e})")

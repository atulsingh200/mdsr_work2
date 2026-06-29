# Hard-Negative Dataset Analysis

Model: `best_mlp_bge_small` (BGE-small + MLP). Threshold 0.5. n=5656 (verified 1:1 row-aligned with predictions).

## Headline

- Overall acc **0.7896** (vs 0.857 on the original eval set).
- Positives acc **0.8649**; hard negatives acc **0.7143** (error **0.2857**) — the entire drop is false positives on hard negs.

## Hypothesis: high sim(y,y*) fools the model → **REJECTED (opposite)**

- corr(sim_y_ystar, error) = **-0.16**, corr(sim_y_ystar, prob) = -0.16 (higher similarity → fewer errors).
- avg sim_y_ystar: fooled(pred=1) = **0.373** vs rejected(pred=0) = **0.423**.

Error by sim_y_ystar quintile:

| sim quintile | n | error | avg prob |
|---|---|---|---|
| [0.010,0.286] | 563 | 0.373 | 0.371 |
| [0.286,0.350] | 561 | 0.332 | 0.331 |
| [0.350,0.446] | 572 | 0.276 | 0.273 |
| [0.446,0.522] | 544 | 0.329 | 0.328 |
| [0.522,0.826] | 588 | 0.328 | 0.130 |

## Real driver: tier gap (reversed-direction distance)

corr(tier_gap, error) = -0.10; corr(tier_gap, sim) = +0.37 (high sim co-occurs with large gaps → generic hub y*, easy to reject).

| tier gap | n | error | mean sim |
|---|---|---|---|
| 1 | 769 | 0.307 | 0.371 |
| 2 | 752 | 0.326 | 0.370 |
| 3 | 495 | 0.164 | 0.400 |
| 4 | 261 | 0.548 | 0.418 |
| 5 | 310 | 0.265 | 0.520 |
| 6 | 53 | 0.132 | 0.422 |
| 7 | 82 | 0.134 | 0.533 |
| 8 | 34 | 0.029 | 0.484 |
| 9 | 21 | 0.095 | 0.486 |
| 10 | 38 | 0.000 | 0.596 |
| 11 | 6 | 0.000 | 0.549 |
| 12 | 7 | 0.000 | 0.669 |

## Top 12 highest-sim_y_ystar hard negatives

| sim | prob | result | dir | sub | x (truncated) | y* (truncated) |
|---|---|---|---|---|---|---|
| 0.752 | 0.001 | rejected | t3→t1 | 14b→2a | Documentation Journey Optimizer Journey Optimizer Tutorials  | We also include Adobe Experience Manager Assets Essentials,  |
| 0.752 | 0.022 | rejected | t3→t1 | 14a→2a | You can preview the field group by clicking the icon. I see  | We also include Adobe Experience Manager Assets Essentials,  |
| 0.752 | 0.977 | FOOLED | t4→t1 | 8c→2a | Okay, here we go. I’ve received the email. Let me show you a | We also include Adobe Experience Manager Assets Essentials,  |
| 0.752 | 0.000 | rejected | t4→t1 | 8b→2a | The ability to evaluate audiences on demand is for these urg | We also include Adobe Experience Manager Assets Essentials,  |
| 0.752 | 0.000 | rejected | t5→t1 | 10d→2a | In this case, let’s look for profile attribute that we want  | We also include Adobe Experience Manager Assets Essentials,  |
| 0.752 | 0.879 | FOOLED | t5→t1 | 10f→2a | Now our results have come back and I have several to choose  | We also include Adobe Experience Manager Assets Essentials,  |
| 0.752 | 0.250 | rejected | t5→t1 | 10g→2a | Documentation Journey Optimizer Journey Optimizer Tutorials  | We also include Adobe Experience Manager Assets Essentials,  |
| 0.752 | 0.000 | rejected | t6→t1 | 9k→2a | Documentation Journey Optimizer Journey Optimizer Tutorials  | We also include Adobe Experience Manager Assets Essentials,  |
| 0.752 | 0.000 | rejected | t6→t1 | 4c→2a | You have to design each kind of field yourself, but since it | We also include Adobe Experience Manager Assets Essentials,  |
| 0.752 | 0.293 | rejected | t6→t1 | 9f→2a | In this video, you’ll learn how to Author an In-app message. | We also include Adobe Experience Manager Assets Essentials,  |
| 0.752 | 0.688 | FOOLED | t6→t1 | 9f→2a | Documentation Journey Optimizer Journey Optimizer Tutorials  | We also include Adobe Experience Manager Assets Essentials,  |
| 0.752 | 0.000 | rejected | t6→t1 | 4a→2a | First, give your campaign a name. You can also add a descrip | We also include Adobe Experience Manager Assets Essentials,  |

Full sorted list: `analysis/hard_neg_top_similarity.csv`. Plots: `analysis/plots/hardneg_error_vs_sim.png`, `analysis/plots/hardneg_error_vs_gap.png`.

**Conclusion:** the model is not fooled by content similarity of y/y*; it fails on small reversed tier gaps (weak directionality) and a few topic pairs (gap=4: 10f/10c/10d→2a). It leans toward predicting causal=1.

# Workflow Step-Ordering Baseline — Results

Task and metrics follow arXiv:2511.04688v2 (inference-only; the paper did NOT fine-tune).
Eval set: 697 Adobe workflow items (2-4 steps). Note step counts are much shorter than
the paper's recipe domain (6-8 steps), so absolute numbers are not directly comparable —
the paper column is shown only for trend/ranking context.

Acc / NLCS / KTau higher = better; NED lower = better.

| Model | Shots | Acc | NLCS | KTau | NED | Unparsed | (paper: Acc/NLCS/KTau/NED) |
|---|---|---|---|---|---|---|---|
| Mistral-7B-Instruct-v0.2 | 0 | 0.504 | 0.756 | 0.312 | 0.433 | 0.168 | 0.29/0.61/0.73/0.55 |
| Mistral-7B-Instruct-v0.2 | 3 | 0.293 | 0.644 | -0.114 | 0.621 | 0.017 | 0.32/0.66/0.79/0.51 |
| Mistral-7B-Instruct-v0.2 | 5 | 0.305 | 0.656 | -0.086 | 0.609 | 0.036 | 0.31/0.66/0.79/0.51 |
| Qwen2.5-7B-Instruct | 0 | 0.825 | 0.906 | 0.739 | 0.159 | 0.001 | 0.71/0.88/0.92/0.22 |
| Qwen2.5-7B-Instruct | 3 | 0.918 | 0.960 | 0.855 | 0.078 | 0.000 | 0.63/0.82/0.88/0.3 |
| Qwen2.5-7B-Instruct | 5 | 0.933 | 0.968 | 0.884 | 0.063 | 0.000 | 0.62/0.81/0.87/0.3 |

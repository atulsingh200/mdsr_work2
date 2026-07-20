## Table B — per-event O(n) vs pairwise O(n^2)

| method | relation-F1 | precision | recall | forwards/doc | total forwards | speedup |
|---|---|---|---|---|---|---|
| LLMERE per-event O(n) (Qwen-7B) | 0.7653 | 0.7894 | 0.7427 | O(n) | 2057 | — |
| pairwise O(n^2) (same docs) | — | — | — | O(n^2) | 5845 | 2.84x more forwards |

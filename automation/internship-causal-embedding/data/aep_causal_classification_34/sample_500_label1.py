import json
import random

INPUT = "directional_train.jsonl"
OUTPUT = "directional_train_sample500_label1.jsonl"
SEED = 42
N = 500

random.seed(SEED)

with open(INPUT) as f:
    label1 = [line for line in f if json.loads(line).get("label") == 1]

sample = random.sample(label1, N)

with open(OUTPUT, "w") as f:
    f.writelines(sample)

print(f"Wrote {N} samples to {OUTPUT}")

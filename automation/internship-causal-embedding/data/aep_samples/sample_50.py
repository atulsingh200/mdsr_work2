import json
import random

input_path = "directional_train.jsonl"
output_path = "directional_train_sample50.jsonl"
n = 50
seed = 42

with open(input_path) as f:
    lines = f.readlines()

random.seed(seed)
sampled = random.sample(lines, min(n, len(lines)))

with open(output_path, "w") as f:
    f.writelines(sampled)

print(f"Sampled {len(sampled)} / {len(lines)} records → {output_path}")

"""LLM-assisted cycle breaker — faithful reproduction of paper 2's Appendix-A prompt.

Given a procedure's steps and a set of predicted "before" relations that form a contradiction
(a cycle), the LLM is asked to name the single relation most likely to be WRONG (to remove). We
present the cycle in DOT format exactly as the paper does, zero-shot, with no few-shot exemplars and
no prompt optimization. Uses a local non-fine-tuned model (Qwen2.5-14B-Instruct) via vLLM.

This is the "you still need an LLM at inference" competitor. The paper's own finding (their Table 3)
is that this underperforms the simple confidence heuristic.
"""

from __future__ import annotations

import random
import re
import string

_SYS = (
    "You are an expert at temporal ordering of the steps in a procedure. "
    "You are given the steps of a procedure and a set of 'before' relations between them, written as "
    "a directed graph in DOT format, where an edge  i -> j  means the model predicted that step i "
    "happens before step j. These relations contain a contradiction: following the arrows leads back "
    "to where you started (a cycle), which is impossible for a real timeline. "
    "Your job is to identify the SINGLE relation (edge) that is most likely WRONG and should be "
    "removed to fix the contradiction. Respond with only that edge in the form  i -> j ."
)


def _anonymize(cycle_edges, steps, rng):
    """Map involved node ids to shuffled anonymous labels so the LLM cannot read gold order off the
    indices. Steps are listed in a shuffled order; DOT uses the anonymous labels. Returns
    (prompt_text, label2node)."""
    involved = list({n for e in cycle_edges for n in e})
    rng.shuffle(involved)                                   # random presentation order
    labels = list(string.ascii_uppercase)
    node2label = {n: labels[k] for k, n in enumerate(involved)}
    label2node = {v: k for k, v in node2label.items()}

    lines = ["Procedure steps (unordered):"]
    for n in involved:                                      # shuffled order, anonymous labels
        lines.append(f"[{node2label[n]}] {steps[n]}")
    dot_edges = [(node2label[u], node2label[v]) for u, v in cycle_edges]
    rng.shuffle(dot_edges)
    dot = "digraph {\n" + "\n".join(f"  {u} -> {v};" for u, v in dot_edges) + "\n}"
    lines.append("\nThe following 'before' relations form a contradiction (cycle):")
    lines.append(dot)
    lines.append("\nWhich single edge is most likely incorrect and should be removed? "
                 "Answer with only the edge, e.g.  X -> Y .")
    return "\n".join(lines), label2node


class LLMBreaker:
    def __init__(self, model="Qwen/Qwen2.5-14B-Instruct", tp=1, gpu_mem_util=0.90,
                 max_model_len=4096):
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=model, tensor_parallel_size=tp, gpu_memory_utilization=gpu_mem_util,
                       max_model_len=max_model_len, dtype="bfloat16", trust_remote_code=True)
        self.tok = self.llm.get_tokenizer()
        self.sp = SamplingParams(temperature=0.0, max_tokens=16)
        self.model = model

    def choose(self, cycle_edges, steps):
        """Return the (u, v) edge the LLM wants to remove, or None if unparseable.

        Node labels are anonymized (shuffled letters) so the LLM cannot infer gold order from
        indices; it must reason from the step text. The chosen anonymous edge is mapped back.
        """
        rng = random.Random(hash(tuple(sorted(cycle_edges))) & 0xFFFFFFFF)
        prompt, label2node = _anonymize(cycle_edges, steps, rng)
        msgs = [{"role": "system", "content": _SYS}, {"role": "user", "content": prompt}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        out = self.llm.generate([text], self.sp, use_tqdm=False)[0].outputs[0].text
        return self._parse(out, cycle_edges, label2node)

    @staticmethod
    def _parse(gen, cycle_edges, label2node):
        m = re.search(r"([A-Z])\s*-+>\s*([A-Z])", gen.upper())
        if m:
            a, b = m.group(1), m.group(2)
            if a in label2node and b in label2node:
                e = (label2node[a], label2node[b])
                if e in cycle_edges:
                    return e
                if (e[1], e[0]) in cycle_edges:
                    return (e[1], e[0])
        # fallback: two labels mentioned in order
        labs = [c for c in re.findall(r"[A-Z]", gen.upper()) if c in label2node]
        if len(labs) >= 2:
            e = (label2node[labs[0]], label2node[labs[1]])
            if e in cycle_edges:
                return e
        return None

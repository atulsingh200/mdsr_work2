"""LLM-assisted cycle breaker using Azure OpenAI (GPT-5-mini).

Same prompt and anonymization logic as llm_breaker.py but calls Azure OpenAI
instead of a local vLLM model.
"""
from __future__ import annotations

import random
import re
import string
from openai import AzureOpenAI

AZURE_DEPLOYMENT_NAME = "gpt-5-mini"
AZURE_ENDPOINT        = "https://gem-apim-stage-va7.azure-api.net/"
AZURE_API_KEY         = "86f1b56331d74c43aa1d8fba23c7eda1"
AZURE_API_VERSION     = "2025-04-01-preview"

_SYS = (
    "Task Overview:\n"
    "You are given the steps of a procedure, in which some steps are uniquely marked by "
    "[STEP#ID]step[/STEP#ID] (e.g., [STEP1]step1[/STEP1], [STEP2]step2[/STEP2]).\n"
    "You are also given a dot graph which represents the chronological order of steps with an error, "
    "where some edges form cycles.\n"
    "Your task is to decide which edge to drop (by its unique_id), being concise and removing the "
    "minimum number of edges.\n"
    "Pay attention: I used a classifier to choose the most fitted relation (label attribute in the "
    "dot graph) and score which represents the confidence of the classifier.\n\n"
    "relation meaning:\n"
    "before - the first step happened before the second.\n"
    "after - the first step happened after the second."
)


def _anonymize(cycle_edges, steps, G, rng):
    """Anonymize steps with shuffled letter labels, include confidence scores in DOT format
    exactly as the paper does (Figure 7 of INLG 2025)."""
    involved = list({n for e in cycle_edges for n in e})
    rng.shuffle(involved)
    labels = list(string.ascii_uppercase)
    node2label = {n: labels[k] for k, n in enumerate(involved)}
    label2node = {v: k for k, v in node2label.items()}

    # Steps section
    lines = ["Procedure steps (unordered):"]
    for n in involved:
        lines.append(f"[{node2label[n]}] {steps[n]}")

    # DOT graph with confidence scores and unique_ids — exactly like paper's Figure 7
    dot_edges = []
    for uid, (u, v) in enumerate(cycle_edges):
        lu, lv = node2label[u], node2label[v]
        # Get confidence from graph edge
        conf = G[u][v].get('confidence', 0.5) if G.has_edge(u, v) else G[v][u].get('confidence', 0.5)
        dot_edges.append((lu, lv, conf, uid))
    rng.shuffle(dot_edges)

    dot_lines = ['digraph Chronology {']
    for lu, lv, conf, uid in dot_edges:
        dot_lines.append(f'    "{lu}" -> "{lv}" [label="BEFORE", score={conf:.8f}, unique_id={uid}];')
    dot_lines.append('}')

    lines.append("\nThe following dot graph represents chronological order with an error "
                 "(some edges form a cycle):")
    lines.append("\n".join(dot_lines))
    lines.append("\nRespond only with the unique_id to drop (the wrong edge).")

    return "\n".join(lines), label2node, {uid: (u, v) for uid, (u, v) in enumerate(cycle_edges)}


class AzureLLMBreaker:
    def __init__(self):
        self.client = AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            api_version=AZURE_API_VERSION,
        )
        self.deployment = AZURE_DEPLOYMENT_NAME
        self.calls = 0

    def choose(self, cycle_edges, steps, G=None):
        rng = random.Random(hash(tuple(sorted(cycle_edges))) & 0xFFFFFFFF)
        prompt, label2node, uid2edge = _anonymize(cycle_edges, steps, G, rng)
        try:
            resp = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": _SYS},
                    {"role": "user",   "content": prompt},
                ],
                max_completion_tokens=1000,  # o-series uses tokens for internal reasoning first
            )
            gen = resp.choices[0].message.content or ""
            self.calls += 1
        except Exception as e:
            print(f"  [Azure API error: {e}]", flush=True)
            return None
        return self._parse(gen, cycle_edges, label2node, uid2edge)

    @staticmethod
    def _parse(gen, cycle_edges, label2node, uid2edge):
        # Primary: parse unique_id integer (paper's method)
        m = re.search(r'\b(\d+)\b', gen)
        if m:
            uid = int(m.group(1))
            if uid in uid2edge:
                return uid2edge[uid]
        # Fallback: parse A -> B letter format
        m = re.search(r"([A-Z])\s*-+>\s*([A-Z])", gen.upper())
        if m:
            a, b = m.group(1), m.group(2)
            if a in label2node and b in label2node:
                e = (label2node[a], label2node[b])
                if e in cycle_edges:
                    return e
                if (e[1], e[0]) in cycle_edges:
                    return (e[1], e[0])
        # Last resort: two letters mentioned
        labs = [c for c in re.findall(r"[A-Z]", gen.upper()) if c in label2node]
        if len(labs) >= 2:
            e = (label2node[labs[0]], label2node[labs[1]])
            if e in cycle_edges:
                return e
        return None

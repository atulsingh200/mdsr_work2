"""FastAPI backend for the CrossEncoder2x scorer UI.

Serves:
  GET  /             -> index.html
  GET  /api/models   -> list of available model names
  POST /api/score    -> forward + reverse scores for a text pair
"""

from __future__ import annotations

import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── paths ──────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_MODEL_CE2X_DIR = Path(
    "/mnt/localssd/new_base_line/internship-causal-embedding"
    "/my_work/kl/cross_encoder_2x"
)
_RUNS_DIR = Path(
    "/mnt/localssd/automation/internship-causal-embedding"
    "/runs/crossencoder2x_deberta"
)

sys.path.insert(0, str(_MODEL_CE2X_DIR))
from model_ce2x import CrossEncoder2x, get_tokenizer  # noqa: E402

# ── model cache ────────────────────────────────────────────────────────────────
_cache: dict[str, dict] = {}   # name -> {"model": ..., "tokenizer": ..., "cfg": ...}
_current_name: Optional[str] = None


def _available_models() -> list[str]:
    """Return model names that have a best.pt checkpoint."""
    names = []
    for d in sorted(_RUNS_DIR.iterdir()):
        if d.is_dir() and (d / "best.pt").exists():
            names.append(d.name)
    return names


def _load_model(name: str) -> dict:
    """Load and cache a model by run-directory name."""
    if name in _cache:
        return _cache[name]

    run_dir = _RUNS_DIR / name
    ckpt_path = run_dir / "best.pt"
    cfg_path = run_dir / "config.json"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"No best.pt found for model '{name}'")

    cfg: dict = {}
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())

    backbone = cfg.get("backbone", "microsoft/deberta-v3-large")
    n_layers = cfg.get("n_layers", 24)
    native = cfg.get("native_backbone", True)
    dropout = cfg.get("dropout", 0.1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CrossEncoder2x(
        backbone=backbone,
        n_layers=n_layers,
        dropout=dropout,
        native_backbone=native,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tokenizer = get_tokenizer(backbone)
    max_seq_len = cfg.get("max_seq_len", 512)

    entry = {
        "model": model,
        "tokenizer": tokenizer,
        "cfg": cfg,
        "device": device,
        "max_seq_len": max_seq_len,
    }
    _cache[name] = entry
    return entry


@torch.no_grad()
def _score_pair(entry: dict, text_a: str, text_b: str) -> float:
    """Return sigmoid(logit) for [CLS] text_a [SEP] text_b [SEP]."""
    tokenizer = entry["tokenizer"]
    model: CrossEncoder2x = entry["model"]
    device = entry["device"]
    max_len = entry["max_seq_len"]

    enc = tokenizer(
        text_a,
        text_b,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        padding=True,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    logit = model(
        enc["input_ids"],
        enc.get("attention_mask"),
        enc.get("token_type_ids"),
    )
    prob = torch.sigmoid(logit).item()
    return prob


# ── FastAPI app ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm-up: nothing required; models are loaded lazily.
    yield


app = FastAPI(title="CrossEncoder2x Scorer", lifespan=lifespan)

# Serve static files (index.html etc.)
app.mount("/static", StaticFiles(directory=str(_HERE / "static")), name="static")


# ── routes ─────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    return FileResponse(str(_HERE / "static" / "index.html"))


class ScoreRequest(BaseModel):
    model_name: str
    text1: str
    text2: str


class ScoreResponse(BaseModel):
    model_name: str
    forward: float   # score(text1 → text2)
    reverse: float   # score(text2 → text1)
    logit_forward: float
    logit_reverse: float


@app.get("/api/models")
def list_models():
    return {"models": _available_models()}


@app.post("/api/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    if not req.text1.strip() or not req.text2.strip():
        raise HTTPException(status_code=422, detail="Both text1 and text2 must be non-empty.")

    try:
        entry = _load_model(req.model_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model load error: {e}")

    try:
        fwd = _score_pair(entry, req.text1, req.text2)
        rev = _score_pair(entry, req.text2, req.text1)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring error: {e}")

    import math
    def to_logit(p: float) -> float:
        p = max(1e-7, min(1 - 1e-7, p))
        return math.log(p / (1 - p))

    return ScoreResponse(
        model_name=req.model_name,
        forward=round(fwd, 6),
        reverse=round(rev, 6),
        logit_forward=round(to_logit(fwd), 4),
        logit_reverse=round(to_logit(rev), 4),
    )


@app.delete("/api/cache/{model_name}")
def evict_cache(model_name: str):
    if model_name in _cache:
        del _cache[model_name]
        return {"evicted": model_name}
    return {"evicted": None}


@app.get("/api/cache")
def cache_status():
    return {"loaded": list(_cache.keys())}

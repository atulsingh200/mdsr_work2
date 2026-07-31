"""CrossEncoder-2x: parameter-matched cross-encoder for fair comparison with BiEncoder.

Architecture
------------
The BiEncoder baseline uses **two untied BERT-base towers** (~218 M parameters total).
To match that budget with a single cross-encoder backbone we need ~2× the transformer
depth.  We achieve this by:

  1. Instantiating a BERT config with `num_hidden_layers=24` (double base depth),
     same hidden_size=768, intermediate_size=3072, num_attention_heads=12.
  2. Initialising every pair of layers from the pretrained bert-base-uncased weights:
        layer  0 ← bert-base layer  0
        layer  1 ← bert-base layer  1
        ...
        layer 11 ← bert-base layer 11
        layer 12 ← bert-base layer  0   (repeated)
        ...
        layer 23 ← bert-base layer 11   (repeated)
     Embeddings and pooler are copied as-is from bert-base.

This gives 194,536,704 trainable parameters — close to the 218,964,480 of the
BiEncoder; the ~24 M gap is exactly the duplicated embedding table that a two-tower
model carries but a single-encoder model shares.

Forward pass (identical to the 1x cross-encoder):
    [CLS] anchor [SEP] candidate [SEP]  →  BERT-24L  →  CLS  →  Linear(1)  →  logit

Train with BCE-with-logits; at eval time use the logit as the similarity score.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, BertConfig, BertModel


def _build_stacked_bert(
    base_name: str = "google-bert/bert-base-uncased",
    n_stacked_layers: int = 24,
) -> BertModel:
    """Build a BERT model with n_stacked_layers, initialised by repeating base layers.

    Each layer i in [0, n_stacked_layers) is copied from the pretrained base layer
    (i % num_base_layers).  Embeddings and pooler are copied from the base model.
    """
    base = AutoModel.from_pretrained(base_name)
    base_cfg = base.config
    n_base = base_cfg.num_hidden_layers

    new_cfg = BertConfig(
        vocab_size=base_cfg.vocab_size,
        hidden_size=base_cfg.hidden_size,
        num_hidden_layers=n_stacked_layers,
        num_attention_heads=base_cfg.num_attention_heads,
        intermediate_size=base_cfg.intermediate_size,
        hidden_act=base_cfg.hidden_act,
        hidden_dropout_prob=base_cfg.hidden_dropout_prob,
        attention_probs_dropout_prob=base_cfg.attention_probs_dropout_prob,
        max_position_embeddings=base_cfg.max_position_embeddings,
        type_vocab_size=base_cfg.type_vocab_size,
        initializer_range=base_cfg.initializer_range,
        layer_norm_eps=getattr(base_cfg, "layer_norm_eps", 1e-12),
        pad_token_id=base_cfg.pad_token_id,
    )
    new_model = BertModel(new_cfg)

    # Copy embeddings and pooler from base
    new_model.embeddings.load_state_dict(base.embeddings.state_dict())
    new_model.pooler.load_state_dict(base.pooler.state_dict())

    # Repeat transformer layers: layer i ← base layer (i % n_base)
    base_layers = base.encoder.layer
    for i, new_layer in enumerate(new_model.encoder.layer):
        src_layer = base_layers[i % n_base]
        new_layer.load_state_dict(src_layer.state_dict())

    return new_model


class CrossEncoder2x(nn.Module):
    """Cross-encoder with a 24-layer BERT backbone (~2× BERT-base parameters).

    Input:  [CLS] anchor [SEP] candidate [SEP]
    Output: scalar logit per pair, shape (B,)

    Set native_backbone=True to use any AutoModel backbone (e.g. DeBERTa-v3-large)
    directly without the BERT-stacking logic.
    """

    def __init__(
        self,
        backbone: str = "google-bert/bert-base-uncased",
        n_layers: int = 24,
        dropout: float = 0.1,
        native_backbone: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.n_layers = n_layers
        if native_backbone:
            # Force fp32: some checkpoints (e.g. deberta-v3-large) ship fp16 weights,
            # which would mismatch the fresh fp32 classifier head. fp32 master weights
            # + autocast (handled in the train loop) is the correct mixed-precision setup.
            self.backbone = AutoModel.from_pretrained(backbone, dtype=torch.float32)
        else:
            self.backbone = _build_stacked_bert(backbone, n_layers)
        hidden = self.backbone.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, 1)

        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits of shape (B,)."""
        kwargs: dict = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        outputs = self.backbone(**kwargs)
        cls = outputs.last_hidden_state[:, 0, :]
        cls = self.dropout(cls)
        return self.classifier(cls).squeeze(-1)


class CrossEncoder2xWithReasoning(nn.Module):
    """CrossEncoder2x + an auxiliary reasoning-decoder head.

    Same encoder/classifier path as CrossEncoder2x (binary directional logit
    off the [CLS] token). In addition, a small Transformer decoder
    cross-attends over the *full* encoder token sequence to reconstruct a
    free-text reasoning trace (teacher-forced), tying its output projection
    to the encoder's input word-embedding table. The decoder is trained
    jointly via an auxiliary token-level cross-entropy loss (combined
    externally with the classification BCE loss) so the shared encoder
    representation is shaped by the reasoning signal, not just the label.
    """

    def __init__(
        self,
        backbone: str = "google-bert/bert-base-uncased",
        n_layers: int = 24,
        dropout: float = 0.1,
        native_backbone: bool = False,
        decoder_layers: int = 4,
        max_reason_len: int = 192,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.n_layers = n_layers
        if native_backbone:
            self.backbone = AutoModel.from_pretrained(backbone, dtype=torch.float32)
        else:
            self.backbone = _build_stacked_bert(backbone, n_layers)
        hidden = self.backbone.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, 1)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

        self.max_reason_len = max_reason_len
        self.reason_pos_emb = nn.Embedding(max_reason_len, hidden)
        nn.init.normal_(self.reason_pos_emb.weight, std=0.02)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden,
            nhead=self.backbone.config.num_attention_heads,
            dim_feedforward=self.backbone.config.intermediate_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.reason_decoder = nn.TransformerDecoder(dec_layer, num_layers=decoder_layers)
        vocab_size = self.backbone.get_input_embeddings().weight.shape[0]
        self.lm_bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
        decoder_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return (cls_logits (B,), decoder_logits (B,T,V) or None)."""
        kwargs: dict = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        enc_out = self.backbone(**kwargs)
        hidden_states = enc_out.last_hidden_state
        cls = hidden_states[:, 0, :]
        cls_logit = self.classifier(self.dropout(cls)).squeeze(-1)

        decoder_logits = None
        if decoder_input_ids is not None:
            embed_w = self.backbone.get_input_embeddings().weight  # (V, H), tied
            seq_len = decoder_input_ids.size(1)
            positions = torch.arange(seq_len, device=decoder_input_ids.device)
            tgt = F.embedding(decoder_input_ids, embed_w) + self.reason_pos_emb(positions)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=tgt.device), diagonal=1
            )
            dec_out = self.reason_decoder(
                tgt,
                hidden_states,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=decoder_padding_mask,
                memory_key_padding_mask=(attention_mask == 0),
            )
            decoder_logits = F.linear(dec_out, embed_w, self.lm_bias)

        return cls_logit, decoder_logits


def get_tokenizer(name: str = "google-bert/bert-base-uncased"):
    return AutoTokenizer.from_pretrained(name)

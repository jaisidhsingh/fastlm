import math
import typing as tp
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from src.models.components import RMSNorm
from src.models.embeddings import apply_rotary_emb_complex_like

try:
  from torch.nn.attention.flex_attention import BlockMask, flex_attention

  _FLEX_ATTENTION_AVAILABLE = True
except ImportError:
  _FLEX_ATTENTION_AVAILABLE = False


class GatedAttention(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    assert cfg.dim % cfg.n_heads == 0
    self.n_heads = cfg.n_heads
    self.head_dim = cfg.dim // cfg.n_heads
    self.dtype = torch.bfloat16 if cfg.model_dtype == 'bfloat16' else torch.float32

    self.w_qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
    self.w_out = nn.Linear(cfg.dim, cfg.dim, bias=False)

    self.use_gate = cfg.attn_gate
    self.qk_norm = cfg.attn_qk_norm
    self.use_flex_attention = getattr(cfg, 'use_flex_attention', False)
    self.dtype = torch.bfloat16

    if self.use_flex_attention:
      if not _FLEX_ATTENTION_AVAILABLE:
        raise ImportError(
          'use_flex_attention=True requires PyTorch >= 2.5. Update PyTorch or set use_flex_attention=False.'
        )
      self._compiled_flex_attention = torch.compile(flex_attention)

    if self.use_gate:
      self.w_gate = nn.Linear(cfg.dim, cfg.dim, bias=False)
    if self.qk_norm:
      self.q_norm = RMSNorm(self.head_dim, cfg.rmsnorm_eps)
      self.k_norm = RMSNorm(self.head_dim, cfg.rmsnorm_eps)

  def forward(self, x, freqs_cis: torch.Tensor | None, attention_mask: torch.Tensor | None = None):
    bsz, seqlen, d = x.shape  # (bsz, seqlen, d)

    q, k, v = self.w_qkv(x).split(d, dim=2)  # (bsz, seqlen, d)
    q = q.view(bsz, seqlen, self.n_heads, self.head_dim)  # (bsz, seqlen, nh, h_dim)
    k = k.view(bsz, seqlen, self.n_heads, self.head_dim)  # (bsz, seqlen, nh, h_dim)
    v = v.view(bsz, seqlen, self.n_heads, self.head_dim).to(dtype=self.dtype)  # (bsz, seqlen, nh, h_dim)

    if self.qk_norm:
      q = self.q_norm(q)
      k = self.k_norm(k)

    if freqs_cis is not None:
      q, k = apply_rotary_emb_complex_like(q, k, freqs_cis=freqs_cis)  # (bsz, seqlen, nh, h_dim)
      q = q.to(dtype=v.dtype)
      k = k.to(dtype=v.dtype)

    q = q.transpose(1, 2)  # (bsz, nh, seqlen, h_dim)
    k = k.transpose(1, 2)  # (bsz, nh, seqlen, h_dim)
    v = v.transpose(1, 2)  # (bsz, nh, seqlen, h_dim)

    # different attention-implementations for different setups
    if attention_mask is not None and isinstance(attention_mask, BlockMask):
      out = self._compiled_flex_attention(q, k, v, block_mask=attention_mask)  # (bsz, nh, seqlen, h_dim)

    else:
      with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (bsz, nh, seqlen, h_dim)

    out = out.transpose(1, 2).contiguous().view(bsz, seqlen, d)  # (bsz, seqlen, d)

    if self.use_gate:
      out = out * torch.sigmoid(self.w_gate(x))
    return self.w_out(out)

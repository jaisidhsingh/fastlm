import math
import typing as tp
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from fla.layers import GatedDeltaNet
from src.models.components import GLU, MLP, MLPReluSquared, RMSNorm
from src.models.embeddings import apply_rotary_emb_complex_like, precompute_freqs_cis


@dataclass
class ModelConfig:
  vocab_size: int
  seq_len: int
  dim: int
  expand: float
  n_layers: int
  n_heads: int
  mlp: str = 'mlp'
  rmsnorm_eps: float = 1e-6
  tie_embeddings: bool = False

  # if there is not `+` symbol in the string below, the model is instantiated as a pure (non-hybrid) model
  token_mixer: str = 'gdn+attn'
  hybrid_mixer_ratio: int = 3
  # means 3:1 (one attention after every 3 gdn layers, type before the + symbol is the "every ratio layers")
  token_mixer_pattern: str = None
  # use this for more fine-grained control over layer arrangement
  layer_norm_scaling: bool = False
  residual_connection: str = 'add'
  attn_gate: bool = True
  attn_qk_norm: bool = True

  gdn_conv_size: int = 4
  gdn_gate: bool = True
  gdn_neg_eigval: bool = True


MLP_CLASSES = {'mlp': MLP, 'glu': GLU, 'mlp_relu_sq': MLPReluSquared}


class GatedAttention(nn.Module):
  def __init__(self, cfg: ModelConfig):
    super().__init__()
    assert cfg.dim % cfg.n_heads == 0
    self.n_heads = cfg.n_heads
    self.head_dim = cfg.dim // cfg.n_heads

    self.w_qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
    self.w_out = nn.Linear(cfg.dim, cfg.dim, bias=False)

    self.use_gate = cfg.attn_gate
    self.qk_norm = cfg.attn_qk_norm

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
    v = v.view(bsz, seqlen, self.n_heads, self.head_dim)  # (bsz, seqlen, nh, h_dim)

    if self.qk_norm:
      q = self.q_norm(q)
      k = self.k_norm(k)

    if freqs_cis is not None:
      q, k = apply_rotary_emb_complex_like(q, k, freqs_cis=freqs_cis)  # (bsz, seqlen, nh, h_dim)

    q = q.transpose(1, 2)  # (bsz, nh, seqlen, h_dim)
    k = k.transpose(1, 2)  # (bsz, nh, seqlen, h_dim)
    v = v.transpose(1, 2)  # (bsz, nh, seqlen, h_dim)

    if attention_mask is not None:
      # attn_mask has shape (bsz, seqlen, seqlen)
      # from (bsz, L, L) to (bsz, 1, L, L) so it broadcasts over heads

      # import pdb
      # pdb.set_trace()
      attention_mask = attention_mask.unsqueeze(1)
      # pdb.set_trace()

      out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)  # (bsz, nh, seqlen, h_dim)
    else:
      out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (bsz, nh, seqlen, h_dim)

    out = out.transpose(1, 2).contiguous().view(bsz, seqlen, d)  # (bsz, seqlen, d)
    if self.use_gate:
      gating = torch.sigmoid(self.w_gate(x))
      out = out * gating

    return self.w_out(out)


SUPPORTED_TOKEN_MIXERS = {'gdn': GatedDeltaNet, 'attn': GatedAttention}


class Block(nn.Module):
  def __init__(self, layer_id: int, token_mixer_type: str, cfg: ModelConfig):
    super().__init__()
    assert token_mixer_type in SUPPORTED_TOKEN_MIXERS, 'Input token mixer is not supported'

    if token_mixer_type == 'attn':
      self.token_mixer = GatedAttention(cfg)
    elif token_mixer_type == 'gdn':
      self.token_mixer = GatedDeltaNet(
        hidden_size=cfg.dim,
        num_heads=cfg.n_heads,
        head_dim=cfg.dim // cfg.n_heads,
        allow_neg_eigval=cfg.gdn_neg_eigval,
        use_gate=cfg.gdn_gate,
        conv_size=cfg.gdn_conv_size,
      )
    self.mlp = MLP_CLASSES[cfg.mlp](dim=cfg.dim, hidden_dim=int(cfg.expand * cfg.dim))

    self.layer_id = layer_id
    self.residual_connection = cfg.residual_connection

    if self.residual_connection == 'add':
      self.token_mixer_norm = RMSNorm(cfg.dim, cfg.rmsnorm_eps)
      self.mlp_norm = RMSNorm(cfg.dim, cfg.rmsnorm_eps)
      self.layer_norm_scaling = cfg.layer_norm_scaling

    if self.residual_connection != 'add':
      self.attn_mhc = lambda x: x
      self.mlp_mhc = lambda x: x

  def forward(self, x, freqs_cis, attention_mask):
    # x: (bsz, seqlen, dim)
    if self.residual_connection == 'add':
      scaling = 1.0 if not self.layer_norm_scaling else 1 / math.sqrt(self.layer_id + 1)

      o = None
      if isinstance(self.token_mixer, GatedAttention):
        o = self.token_mixer(scaling * self.token_mixer_norm(x), freqs_cis, attention_mask)

      elif isinstance(self.token_mixer, GatedDeltaNet):
        if attention_mask.ndim == 3:
          o, _, past_key_values = self.token_mixer(
            hidden_states=scaling * self.token_mixer_norm(x), attention_mask=attention_mask[:, :, 0]
          )
        elif attention_mask.ndim == 2:
          o, _, past_key_values = self.token_mixer(
            hidden_states=scaling * self.token_mixer_norm(x), attention_mask=attention_mask
          )

      else:
        raise NotImplementedError('Unsupported value encountered for `cfg.token_mixer`')

      x = x + o
      x = x + self.mlp(scaling * self.mlp_norm(x))

    # not supported rn
    elif self.residual_connection == 'mhc':
      raise NotImplementedError('Manifold-Constrained HyperConnections are currently not supported.')

    return x


class Transformer(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.n_layers = cfg.n_layers
    head_dim = cfg.dim // cfg.n_heads
    if cfg.dim % cfg.n_heads != 0:
      raise ValueError('dim must be divisible by n_heads')

    self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.dim)
    self.layers = self._prepare_layers(cfg)
    self.out_norm = RMSNorm(cfg.dim, cfg.rmsnorm_eps)
    self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

    self.freqs_cis = precompute_freqs_cis(head_dim, cfg.seq_len, 500000)[0 : cfg.seq_len]

    # init all weights, scale residual branches
    self.apply(self._init_weights)
    self._scale_residual_branches()

    if cfg.tie_embeddings:
      self.tie_weights()

  def _prepare_layers(self, cfg):
    if cfg.token_mixer_pattern is None:
      if '+' in cfg.token_mixer:
        token_mixers = cfg.token_mixer.split('+')
        assert len(token_mixers) == 2, 'Only support two token mixers for now'
        assert all(tm in SUPPORTED_TOKEN_MIXERS for tm in token_mixers), (
          f'Unknown token mixer(s) {token_mixers}, supported: {list(SUPPORTED_TOKEN_MIXERS.keys())}'
        )
        layers = []
        for idx in range(cfg.n_layers):
          if (idx + 1) % (cfg.hybrid_mixer_ratio + 1) == 0:
            token_mixer_type = token_mixers[-1]
          else:
            token_mixer_type = token_mixers[0]
          layers.append(Block(idx, token_mixer_type, cfg))

      else:
        layers = []
        for idx in range(cfg.n_layers):
          layers.append(Block(idx, cfg.token_mixer, cfg))

    else:
      # here's the pattern we like: "(gdn,gdn,gdn,attn)..."
      # the "..." means repeated enough times till `n_layers` is reached
      # the number of comma-separated modes in the parenthesis should be perfectly divide `n_layers`
      block_pattern = cfg.token_mixer_pattern.split('...')[0][1:-1].split(',')
      assert cfg.n_layers % len(block_pattern) == 0, (
        'Size of a block in cfg.token_mixer_pattern must perfectly divide cfg.n_layers'
      )
      num_repeats = int(cfg.n_layers / len(block_pattern))

      layers = []
      idx = 0
      for i in range(num_repeats):
        for token_mixer_type in block_pattern:
          layers.append(Block(idx, token_mixer_type, cfg))
          idx += 1

      assert idx == cfg.n_layers, 'Error in creating layers from cfg.token_mixer_pattern'

    return nn.ModuleList(layers)

  def forward(self, x, attention_mask):
    # x: (bsz, seqlen)
    x = self.embed_tokens(x)  # (bsz, seqlen, dim)
    self.freqs_cis = self.freqs_cis.to(x.device)

    for layer in self.layers:
      x = layer(x, self.freqs_cis, attention_mask)  # (bsz, seqlen, dim)

    return self.lm_head(self.out_norm(x))  # (bsz, seqlen, vocab_size)

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def _scale_residual_branches(self):
    for n, p in self.named_parameters():
      if n.endswith('fc2.weight'):  # mlp/glu output layer
        torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layers))
      if n.endswith('w_out.weight'):  # attn output layer
        torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layers))
      if n.endswith('o_proj.weight'):  # gdn output layer
        torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layers))

  def tie_weights(self):
    self.lm_head.weight = self.embed_tokens.weight

  def count_params(self, non_embedding=True):
    n_params = sum(p.numel() for p in self.parameters())
    if non_embedding:
      n_params -= self.embed_tokens.weight.numel()
      if self.lm_head.weight is not self.embed_tokens.weight:  # if no weight tying
        n_params -= self.lm_head.weight.numel()
    return n_params

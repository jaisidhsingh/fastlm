import math
import typing as tp
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from fla.layers import GatedDeltaNet
from src.models.attention import GatedAttention
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
  model_dtype: str = 'bfloat16'

  # if there is not `+` symbol in the string below, the model is instantiated as a pure (non-hybrid) model
  token_mixer: str = 'gdn+attn'
  # means 3:1 (one attention after every 3 gdn layers, type before the + symbol is the "every ratio layers")
  hybrid_mixer_ratio: int = 3
  layer_norm_scaling: bool = False
  residual_connection: str = 'add'
  attn_gate: bool = True
  attn_qk_norm: bool = True

  gdn_conv_size: int = 4
  gdn_gate: bool = True
  gdn_neg_eigval: bool = True

  intra_doc: bool = False
  use_flex_attention: bool = False


MLP_CLASSES = {'mlp': MLP, 'glu': GLU, 'mlp_relu_sq': MLPReluSquared}
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
        intra_doc=cfg.intra_doc,
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

  def forward(self, x, freqs_cis, attention_mask, linear_mask, cu_seqlens):
    # x: (bsz, seqlen, dim)
    if self.residual_connection == 'add':
      scaling = 1.0 if not self.layer_norm_scaling else 1 / math.sqrt(self.layer_id + 1)

      o = None
      if isinstance(self.token_mixer, GatedAttention):
        o = self.token_mixer(scaling * self.token_mixer_norm(x), freqs_cis, attention_mask)

      elif isinstance(self.token_mixer, GatedDeltaNet):
        o, _, past_key_values = self.token_mixer(
          hidden_states=scaling * self.token_mixer_norm(x), attention_mask=linear_mask, cu_seqlens=cu_seqlens
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
    self.cfg = cfg
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

  def verify_arch(self):
    if '+' not in self.cfg.token_mixer and self.hybrid_mixer_ratio == 1:
      print(type(self.layers[0].token_mixer).__name__)
    else:
      print(type(self.layers[0].token_mixer).__name__)
      print(type(self.layers[self.hybrid_mixer_ratio].token_mixer).__name__)

  def _prepare_layers(self, cfg):
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

    return nn.ModuleList(layers)

  def forward(self, x, attention_mask=None, linear_mask=None, cu_seqlens=None):
    # x: (bsz, seqlen)
    x = self.embed_tokens(x)  # (bsz, seqlen, dim)
    self.freqs_cis = self.freqs_cis.to(x.device)

    for layer in self.layers:
      x = layer(x, self.freqs_cis, attention_mask, linear_mask, cu_seqlens)  # (bsz, seqlen, dim)

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

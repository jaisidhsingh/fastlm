from fractions import Fraction
from types import SimpleNamespace

from fla.models import GatedDeltaNetConfig, TransformerConfig
from src.models.legacy.transformer import ModelConfig

CONFIG_MAP = {'attn': TransformerConfig, 'gdn': GatedDeltaNetConfig}
# `Transformer` builds its rotary embeddings with `precompute_freqs_cis(head_dim, seq_len, 500000)`.
ROPE_THETA = 500000
# `GLU` and `MLPReluSquared` take this as the `multiple_of` default.
MLP_MULTIPLE_OF = 256


def get_intermediate_size(cfg: SimpleNamespace) -> int:
  """Size the FFN hidden dimension the way the legacy MLP sizes it.

  `Block` passes `int(cfg.expand * cfg.dim)` to the legacy MLP, which then rounds
  it up to a multiple of 256. FLA takes `intermediate_size` at face value, so the
  rounding has to be applied here or the two backends build different FFNs
  whenever `d_model * expand` is not already a multiple of 256.
  """
  hidden_dim = int(cfg.d_model * float(Fraction(cfg.expand)))
  return MLP_MULTIPLE_OF * ((hidden_dim + MLP_MULTIPLE_OF - 1) // MLP_MULTIPLE_OF)


def parse_arch_id(arch_id: str):
  """
  - pure attention corresponds to `arch_id = "attn"`
  - hybrid with gdn:attn = x:1 (one attn layer after x gdn layers)
    corresponds to `arch_id: "gdn+attn_x-1"`
    x is stored as `ratio = x`
  - hybrid with attn:gdn = x:1 (one gdn layer after x attn layers)
    correspoinds to `arch_id: gdn+attn_1-x`
    x is stored as `ratio = -x` (negative indicates reverse layer order)
  """
  split_id = arch_id.split('_')
  arch = split_id[0]
  ratio = None
  if len(split_id) == 2:
    [r1, r2] = [int(x) for x in split_id[1].split('-')]
    if r2 == 1:
      ratio = r1
    elif r1 == 1:
      ratio = -r2
  return arch, ratio


def build_hybrid_layers(n_layers, ratio):
  layers = []
  for i in range(n_layers):
    if ratio > 0:  # means repeat [(r gdn layers), attn]
      if (i + 1) % (ratio + 1) == 0:
        layers.append(i)
    else:  # means repeat [gdn, (r attn layers)]
      r = abs(ratio)
      if i % (r + 1) != 0:
        layers.append(i)

  return layers


def build_kwargs(cfg: SimpleNamespace, arch: str):
  """Map the legacy config fields onto the FLA config fields for one architecture.

  `arch` comes from `parse_arch_id`, so a hybrid gives `"gdn+attn"` and takes the
  `gdn` branch; its attention settings travel in the `attn` dict instead. Only a
  pure `"attn"` architecture puts attention settings at the top level.
  """
  values = vars(cfg)
  kwargs = {
    'norm_eps': values.get('rmsnorm_eps', 1e-6),
    'tie_word_embeddings': values.get('tie_embeddings', False),
  }
  if 'gdn' in arch:
    kwargs['expand_v'] = values.get('expand_v', 2)
    kwargs['head_dim'] = cfg.d_model // cfg.n_heads
    kwargs['conv_size'] = cfg.gdn_conv_size
    kwargs['use_gate'] = cfg.gdn_gate
    kwargs['allow_neg_eigval'] = cfg.gdn_neg_eigval
    kwargs['intra_doc'] = values.get('intra_doc_masking', False)
  elif arch == 'attn':
    kwargs['num_kv_heads'] = cfg.n_heads
    kwargs['qkv_bias'] = False
    kwargs['qk_norm'] = cfg.attn_qk_norm
    kwargs['use_gate'] = cfg.attn_gate
    kwargs['window_size'] = None
    kwargs['rope_theta'] = ROPE_THETA
  return kwargs


def get_hybrid_model_config(cfg: SimpleNamespace, arch: str, ratio: int):
  kwargs = build_kwargs(cfg, arch)
  attn_config_to_insert = dict(
    layers=build_hybrid_layers(cfg.n_layers, ratio),
    hidden_size=cfg.d_model,
    num_heads=cfg.n_heads,
    num_kv_heads=cfg.n_heads,
    qkv_bias=False,
    qk_norm=cfg.attn_qk_norm,
    use_gate=cfg.attn_gate,
    window_size=None,
    rope_theta=ROPE_THETA,
  )
  # `attn` must be passed to the constructor, not assigned afterwards: the validation
  # that fills in the keys `GatedDeltaNetBlock` reads runs only inside `__init__`.
  return GatedDeltaNetConfig(
    hidden_size=cfg.d_model,
    num_heads=cfg.n_heads,
    num_hidden_layers=cfg.n_layers,
    intermediate_size=get_intermediate_size(cfg),
    max_position_embeddings=cfg.seq_len,
    vocab_size=cfg.vocab_size,
    attn=attn_config_to_insert,
    **kwargs,
  )


def get_pure_model_config(cfg: SimpleNamespace, arch: str):
  if arch == 'attn':
    ref = TransformerConfig
  elif arch == 'gdn':
    ref = GatedDeltaNetConfig
  else:
    raise NotImplementedError('Unsupported value of `arch` provided')

  kwargs = build_kwargs(cfg, arch)
  return ref(
    hidden_size=cfg.d_model,
    num_heads=cfg.n_heads,
    num_hidden_layers=cfg.n_layers,
    intermediate_size=get_intermediate_size(cfg),
    max_position_embeddings=cfg.seq_len,
    vocab_size=cfg.vocab_size,
    **kwargs,
  )


def config_builder(cfg):
  arch, ratio = parse_arch_id(cfg.arch_id)

  if ratio is not None:
    model_config = get_hybrid_model_config(cfg, arch, ratio)
  else:
    model_config = get_pure_model_config(cfg, arch)

  return model_config


def _construct_custom_config_for_legacy_backend(cfg):
  """
  Just as a reference for the code above.
  """
  model_cfg = ModelConfig(
    model_dtype=cfg.dtype,
    vocab_size=cfg.vocab_size,
    dim=cfg.d_model,
    expand=float(Fraction(cfg.expand)),
    n_layers=cfg.n_layers,
    n_heads=cfg.n_heads,
    rmsnorm_eps=1e-6,
    mlp=cfg.mlp_class,
    seq_len=cfg.seq_len,
    tie_embeddings=cfg.tie_embeddings,
    token_mixer=cfg.token_mixer,
    hybrid_mixer_ratio=cfg.hybrid_mixer_ratio,
    layer_norm_scaling=cfg.layer_norm_scaling,
    residual_connection=cfg.residual_connection,
    attn_gate=cfg.attn_gate,
    attn_qk_norm=cfg.attn_qk_norm,
    gdn_conv_size=cfg.gdn_conv_size,
    gdn_gate=cfg.gdn_gate,
    gdn_neg_eigval=cfg.gdn_neg_eigval,
    intra_doc=cfg.intra_doc_masking,
    use_flex_attention=getattr(cfg, 'use_flex_attention', True),
  )
  return model_cfg

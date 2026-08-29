from fractions import Fraction
from types import SimpleNamespace
from fla.models import GatedDeltaNetConfig, TransformerConfig
from src.models.legacy.transformer import ModelConfig


CONFIG_MAP = {
  "attn": TransformerConfig,
  "gdn": GatedDeltaNetConfig
}


def parse_arch_id(arch_id: str):
  """
  - pure attention corresponds to `arch_id = "attn"`
  - hybrid with gdn:attn = x:1 corresponds to `arch_id: "gdn+attn_x"`
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
    if ratio > 0: # means repeat [(r x gdn), attn]
      if (i+1) % (ratio+1) == 0:
        layers.append(i)
    else: # means repeat [(r x attn), gdn]
      r = abs(ratio)
      if (i+1) % (r+1) != 0:
        layers.append(i)

  return layers


def build_kwargs(cfg: SimpleNamespace, arch: str):
  kwargs = {}
  if "gdn" in arch:
    kwargs["expand_v"] = vars(cfg).get("expand_v", 2)
  return kwargs


def get_hybrid_model_config(cfg: SimpleNamespace, arch: str, ratio: int):
  kwargs = build_kwargs(cfg, arch)
  config = GatedDeltaNetConfig(
    hidden_size=cfg.d_model,
    num_heads=cfg.n_heads,
    num_hidden_layers=cfg.n_layers,
    intermediate_size=int(cfg.d_model * float(Fraction(cfg.expand))),
    max_position_embeddings=cfg.seq_len,
    vocab_size=cfg.vocab_size,
    **kwargs,
  )

  attn_config_to_insert = dict(
    layers=build_hybrid_layers(cfg.n_layers, ratio),
    hidden_size=cfg.d_model,
    num_heads=cfg.n_heads,
    num_kv_heads=cfg.n_heads,
    qk_norm=cfg.attn_qk_norm,
    use_gate=cfg.attn_gate,
  )
  config.attn = attn_config_to_insert
  return config 


def get_pure_model_config(cfg: SimpleNamespace, arch: str):
  if arch == 'attn':
    ref = TransformerConfig
  elif arch == 'gdn':
    ref = GatedDeltaNetConfig
  else:
    raise NotImplementedError("Unsupported value of `arch` provided")

  kwargs = build_kwargs(cfg, arch)
  return ref(
    hidden_size=cfg.d_model,
    num_heads=cfg.n_heads,
    num_hidden_layers=cfg.n_layers,
    intermediate_size=int(cfg.d_model * float(Fraction(cfg.expand))),
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
  only here as a reference for the above code
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


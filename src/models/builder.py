from fla.models import GatedDeltaNetConfig, TransformerConfig
from src.models.transformer import ModelConfig


def parse_arch_id(arch_id: str):
  """
  - pure attention corresponds to `arch_id = "attn"`
  - hybrid with gdn:attn = x:1 corresponds to `arch_id: "gdn+attn_x"`
  """
  split_id = arch_id.split('_')
  arch = split_id[0]
  ratio = None
  if len(split_id) == 2:
    [r1, r2] = split_id[1].split('-')
    if r2 == 1:
      ratio = r1
    elif r1 == 1:
      ratio = -r2
  return arch, ratio


def config_builder(cfg):
  arch, ratio = parse_arch_id(cfg.arch_id)

  if ratio is not None:
    model_config = get_hybrid_model_config(cfg, arch, ratio)
  else:
    model_config = get_pure_model_config(cfg, arch)

  return model_config


def construct_model_config(cfg):
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


from types import SimpleNamespace

from transformers import AutoConfig, AutoModelForCausalLM

from fla.models import GatedDeltaNetConfig, KDAConfig, SignedKDAConfig, TransformerConfig

CONFIG_MAP = {
  'gdn': GatedDeltaNetConfig,
  'attn': TransformerConfig,
  'kda': KDAConfig,
  'signed-kda': SignedKDAConfig,
  'gdn-neg': GatedDeltaNetConfig,
  'kda-neg': KDAConfig,
}


def set_kwargs(cfg: SimpleNamespace, arch: str) -> dict:
  kwargs = {}
  if 'gdn' in arch or 'kda' in arch:
    kwargs['allow_neg_eigval'] = cfg.allow_neg_eigval

  if 'kda' in arch:
    kwargs['lower_bound'] = cfg.lower_bound

  if 'signed-kda' in arch:
    kwargs['gate'] = cfg.gate
    kwargs['allow_neg_eigval'] = cfg.allow_neg_eigval
    kwargs['lower_bound'] = float(cfg.lower_bound)

  return kwargs


def get_pure_model_config(cfg: SimpleNamespace, arch: str):
  ref = CONFIG_MAP[arch]
  kwargs = set_kwargs(cfg, arch)
  return ref(
    hidden_size=cfg.d_model,
    num_heads=cfg.n_heads,
    num_hidden_layers=cfg.n_layers,
    intermediate_size=cfg.d_ffn,
    expand_v=cfg.expand_v,
    head_dim=cfg.head_dim,
    max_position_embeddings=cfg.seq_len,
    vocab_size=cfg.vocab_size,
    **kwargs,
  )


def get_hybrid_model_config(cfg: SimpleNamespace, arch: str, hybrid_ratio: int = 3):
  [base_arch, inter_arch] = arch.split('+')

  ref = CONFIG_MAP[base_arch]
  kwargs = set_kwargs(cfg, base_arch)
  config = ref(
    hidden_size=cfg.d_model,
    num_heads=cfg.n_heads,
    num_hidden_layers=cfg.n_layers,
    intermediate_size=cfg.d_ffn,
    expand_v=cfg.expand_v,
    head_dim=cfg.head_dim,
    max_position_embeddings=cfg.seq_len,
    vocab_size=cfg.vocab_size,
    **kwargs,
  )

  assert inter_arch == 'attn', 'Only gdn/kda->attn hybrid schemes supported as of now.'
  attn_layer_indices = [i - 1 for i in range(hybrid_ratio, cfg.n_layers, hybrid_ratio)]

  config.attn = {
    'layers': attn_layer_indices,
    'num_heads': cfg.n_heads,
    'num_kv_heads': cfg.n_heads,
    'qkv_bias': False,
    'rope_theta': cfg.rope_theta,
    'window_size': None,
  }
  return config


def get_param_groups(model, weight_decay):
  """Create param groups with and withou weight_decay."""

  # filter out parameters that do not require grad
  named_param_dict = {n: p for n, p in model.named_parameters() if p.requires_grad}

  # filter out parameters with names containing 'bias', 'norm', etc
  decay_params_names = [
    n for n, p in model.named_parameters() if not getattr(p, '_no_weight_decay', False)
  ]  # exclude mamba 'A_log', 'D'
  decay_params_names = [n for n in decay_params_names if 'bias' not in n]  # exclude bias
  decay_params_names = [n for n in decay_params_names if 'norm' not in n]  # exclude normalization layers

  decay_params = [p for n, p in named_param_dict.items() if n in decay_params_names]
  no_decay_params = [p for n, p in named_param_dict.items() if n not in decay_params_names]

  # # sanity check
  # no_decay_param_names = [n for n, p in named_param_dict.items() if n not in decay_params_names]
  # print(f"\nParameters with no weight decay:")
  # print(*no_decay_param_names, sep='\n')
  # print(f"\nParameters with weight decay:")
  # print(*decay_params_names, sep='\n')

  param_groups = [
    {'params': decay_params, 'weight_decay': weight_decay},
    {'params': no_decay_params, 'weight_decay': 0.0},
  ]

  return param_groups

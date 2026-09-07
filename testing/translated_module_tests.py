from __future__ import annotations

import math
from collections.abc import Callable
from fractions import Fraction
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForCausalLM

from fla.layers.attn import Attention
from fla.layers.gated_deltanet import GatedDeltaNet
from fla.layers.legacy_gated_deltanet import LegacyGatedDeltaNet
from fla.modules import GatedMLP
from fla.modules import RMSNorm as FLARMSNorm
from src.models import builder
from src.models.legacy.attention import GatedAttention
from src.models.legacy.components import GLU, RMSNorm
from src.models.legacy.embeddings import precompute_freqs_cis
from src.models.legacy.transformer import ModelConfig, Transformer
from src.models.translate_backend import (
  translate_embeddings,
  translate_ffn,
  translate_gated_attention,
  translate_gated_deltanet,
  translate_lm_head,
  translate_model,
  translate_rms_norm,
)

ModuleFactory = Callable[[SimpleNamespace, Any], nn.Module]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


BF16_RTOL = 1e-2
BF16_ATOL = 1e-3


def _relative(difference: float, reference: float) -> float:
  if reference > 0.0:
    return difference / reference
  return 0.0 if difference == 0.0 else float('inf')


def _token_rms(tensor: torch.Tensor) -> float:
  """Average the RMS norm of the token vectors over batch and sequence length."""
  return tensor.float().pow(2).mean(dim=-1).sqrt().mean().item()


def _rms(tensor: torch.Tensor) -> float:
  return tensor.float().pow(2).mean().sqrt().item()


def _hidden_state_metrics(fla_outputs: torch.Tensor, legacy_outputs: torch.Tensor) -> dict[str, object]:
  difference = fla_outputs.float() - legacy_outputs.float()
  difference_rms = _token_rms(difference)
  reference_rms = _token_rms(legacy_outputs)
  difference_max = difference.abs().max().item()
  reference_max = legacy_outputs.float().abs().max().item()

  return {
    'allclose': torch.allclose(fla_outputs, legacy_outputs, rtol=BF16_RTOL, atol=BF16_ATOL),
    'token rms of difference': difference_rms,
    'token rms of legacy output': reference_rms,
    'relative token rms': _relative(difference_rms, reference_rms),
    'max absolute error': difference_max,
    'relative max absolute error': _relative(difference_max, reference_max),
  }


def _tensor_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, object]:
  difference = actual.float() - expected.float()
  reference = expected.float()

  difference_rms = _rms(difference)
  reference_rms = _rms(reference)
  difference_frobenius = torch.linalg.vector_norm(difference).item()
  reference_frobenius = torch.linalg.vector_norm(reference).item()
  difference_max = difference.abs().max().item()
  reference_max = reference.abs().max().item()

  metrics: dict[str, object] = {
    'equal': torch.equal(actual, expected),
    'allclose': torch.allclose(actual, expected, rtol=BF16_RTOL, atol=BF16_ATOL),
    'rms of difference': difference_rms,
    'rms of reference': reference_rms,
    'relative rms': _relative(difference_rms, reference_rms),
    'frobenius norm of difference': difference_frobenius,
    'frobenius norm of reference': reference_frobenius,
    'relative frobenius norm': _relative(difference_frobenius, reference_frobenius),
  }

  if difference.ndim == 2:
    difference_spectral = torch.linalg.matrix_norm(difference, ord=2).item()
    reference_spectral = torch.linalg.matrix_norm(reference, ord=2).item()
    metrics['spectral norm of difference'] = difference_spectral
    metrics['spectral norm of reference'] = reference_spectral
    metrics['relative spectral norm'] = _relative(difference_spectral, reference_spectral)

  metrics['max absolute error'] = difference_max
  metrics['relative max absolute error'] = _relative(difference_max, reference_max)
  return metrics


def _print_metrics(metrics: dict[str, object], indent: str = '') -> None:
  for name, value in metrics.items():
    print(f'{indent}{name}: {value}')


def _legacy_gated_attention(config: SimpleNamespace, fla_config: Any) -> nn.Module:
  return GatedAttention(config)


def _legacy_gdn(config: SimpleNamespace, fla_config: Any) -> nn.Module:
  return LegacyGatedDeltaNet(
    hidden_size=config.dim,
    num_heads=config.n_heads,
    head_dim=config.dim // config.n_heads,
    allow_neg_eigval=config.gdn_neg_eigval,
    use_gate=config.gdn_gate,
    conv_size=config.gdn_conv_size,
    intra_doc=config.intra_doc,
  )


def _legacy_ffn(config: SimpleNamespace, fla_config: Any) -> nn.Module:
  if config.mlp != 'glu':
    raise ValueError(f'FFN translation only supports the legacy GLU, got {config.mlp!r}.')
  return GLU(dim=config.dim, hidden_dim=int(float(Fraction(config.expand)) * config.dim))


def _legacy_embeddings(config: SimpleNamespace, fla_config: Any) -> nn.Module:
  return nn.Embedding(config.vocab_size, config.dim)


def _legacy_rms_norm(config: SimpleNamespace, fla_config: Any) -> nn.Module:
  return RMSNorm(config.dim, eps=config.rmsnorm_eps)


def _legacy_lm_head(config: SimpleNamespace, fla_config: Any) -> nn.Module:
  return nn.Linear(config.dim, config.vocab_size, bias=False)


def _fla_gated_attention(config: SimpleNamespace, fla_config: Any) -> nn.Module:
  return Attention(
    hidden_size=fla_config.hidden_size,
    num_heads=config.n_heads,
    num_kv_heads=config.n_heads,
    qkv_bias=False,
    qk_norm=config.attn_qk_norm,
    use_gate=config.attn_gate,
    window_size=None,
    rope_theta=500000,
    max_position_embeddings=fla_config.max_position_embeddings,
  )


def _fla_gdn(config: SimpleNamespace, fla_config: Any) -> nn.Module:
  """Build the FLA GDN layer from the builder config alone.

  The argument list is the one `GatedDeltaNetBlock.__init__` uses. Reading only
  `fla_config` is what gives the comparison its content: the two layer classes
  are the same implementation, so the test measures whether `config_builder`
  reproduces the arguments legacy `Block` passes.
  """
  return GatedDeltaNet(
    mode=fla_config.attn_mode,
    hidden_size=fla_config.hidden_size,
    expand_v=fla_config.expand_v,
    head_dim=fla_config.head_dim,
    num_heads=fla_config.num_heads,
    num_v_heads=fla_config.num_v_heads,
    use_gate=fla_config.use_gate,
    use_short_conv=fla_config.use_short_conv,
    allow_neg_eigval=fla_config.allow_neg_eigval,
    conv_size=fla_config.conv_size,
    norm_eps=fla_config.gdn_norm_eps,
    intra_doc=fla_config.intra_doc,
  )


def _fla_ffn(config: SimpleNamespace, fla_config: Any) -> nn.Module:
  return GatedMLP(
    hidden_size=fla_config.hidden_size,
    hidden_ratio=fla_config.hidden_ratio,
    intermediate_size=fla_config.intermediate_size,
    hidden_act=fla_config.hidden_act,
    fuse_swiglu=fla_config.fuse_swiglu,
  )


def _fla_embeddings(config: SimpleNamespace, fla_config: Any) -> nn.Module:
  return nn.Embedding(fla_config.vocab_size, fla_config.hidden_size, fla_config.pad_token_id)


def _fla_rms_norm(config: SimpleNamespace, fla_config: Any) -> nn.Module:
  return FLARMSNorm(fla_config.hidden_size, eps=fla_config.norm_eps)


def _fla_lm_head(config: SimpleNamespace, fla_config: Any) -> nn.Module:
  return nn.Linear(fla_config.hidden_size, fla_config.vocab_size, bias=False)


LEGACY_MODULE_SPEC_MAP: dict[str, ModuleFactory] = {
  'gated_attention': _legacy_gated_attention,
  'gdn': _legacy_gdn,
  'ffn': _legacy_ffn,
  'embeddings': _legacy_embeddings,
  'rms_norm': _legacy_rms_norm,
  'lm_head': _legacy_lm_head,
}

FLA_MODULE_SPEC_MAP: dict[str, ModuleFactory] = {
  'gated_attention': _fla_gated_attention,
  'gdn': _fla_gdn,
  'ffn': _fla_ffn,
  'embeddings': _fla_embeddings,
  'rms_norm': _fla_rms_norm,
  'lm_head': _fla_lm_head,
}

TRANSLATE_MODULE_SPEC_MAP: dict[str, Callable[[dict], dict]] = {
  'gated_attention': translate_gated_attention,
  'gdn': translate_gated_deltanet,
  'ffn': translate_ffn,
  'embeddings': translate_embeddings,
  'rms_norm': translate_rms_norm,
  'lm_head': translate_lm_head,
}


def _infer_arch_id(config: SimpleNamespace) -> str:
  token_mixer = getattr(config, 'token_mixer', 'attn')
  if token_mixer in {'attn', 'gdn'}:
    return token_mixer
  ratio = getattr(config, 'hybrid_mixer_ratio', 1)
  if ratio < 0:
    return f'gdn+attn_1-{abs(ratio)}'
  return f'gdn+attn_{ratio}-1'


def _normalize_config(config: dict | SimpleNamespace) -> SimpleNamespace:
  values = dict(config) if isinstance(config, dict) else vars(config).copy()

  if 'd_model' not in values and 'dim' in values:
    values['d_model'] = values['dim']
  if 'dim' not in values and 'd_model' in values:
    values['dim'] = values['d_model']
  if 'mlp_class' not in values and 'mlp' in values:
    values['mlp_class'] = values['mlp']
  if 'mlp' not in values and 'mlp_class' in values:
    values['mlp'] = values['mlp_class']
  if 'dtype' not in values and 'model_dtype' in values:
    values['dtype'] = values['model_dtype']
  if 'model_dtype' not in values and 'dtype' in values:
    values['model_dtype'] = values['dtype']

  values.setdefault('arch_id', _infer_arch_id(SimpleNamespace(**values)))
  values.setdefault('n_layers', 1)
  values.setdefault('seq_len', 16)
  values.setdefault('expand', 4)
  values.setdefault('mlp', 'glu')
  values.setdefault('mlp_class', values['mlp'])
  values.setdefault('rmsnorm_eps', 1e-6)
  values.setdefault('model_dtype', 'bfloat16')
  values.setdefault('dtype', values['model_dtype'])
  values.setdefault('attn_gate', False)
  values.setdefault('attn_qk_norm', False)
  values.setdefault('gdn_conv_size', 4)
  values.setdefault('gdn_gate', True)
  values.setdefault('gdn_neg_eigval', True)
  # The module tests pass no cu_seqlens and no linear_mask, which the intra_doc
  # branch of GatedDeltaNet.forward needs.
  values.setdefault('intra_doc', False)
  values.setdefault('intra_doc_masking', values['intra_doc'])
  return SimpleNamespace(**values)


def _make_inputs(module_spec: str, config: SimpleNamespace, module: nn.Module) -> torch.Tensor:
  parameter = next(module.parameters())
  batch_size = 2
  sequence_length = config.seq_len
  if module_spec == 'embeddings':
    return torch.randint(
      low=0,
      high=config.vocab_size,
      size=(batch_size, sequence_length),
      device=DEVICE,
    )
  return torch.randn(
    batch_size,
    sequence_length,
    config.dim,
    device=DEVICE,
    dtype=parameter.dtype,
  )


def _forward(
  module_spec: str,
  module: nn.Module,
  inputs: torch.Tensor,
  config: SimpleNamespace,
  *,
  fla_backend: bool,
) -> torch.Tensor:
  if module_spec == 'gdn':
    return module(inputs)[0]

  if module_spec != 'gated_attention':
    return module(inputs)

  if fla_backend:
    return module(inputs, permute_rope_qk=True)[0]

  head_dim = config.dim // config.n_heads
  freqs_cis = precompute_freqs_cis(head_dim, inputs.shape[1], theta=500000).to(DEVICE)
  return module(inputs, freqs_cis=freqs_cis)[0]


def _make_legacy_config() -> SimpleNamespace:
  return _normalize_config(
    dict(
      arch_id='attn',
      vocab_size=256,
      dim=64,
      n_heads=4,
      n_layers=1,
      seq_len=16,
      expand=4,
      mlp='glu',
      rmsnorm_eps=1e-6,
      model_dtype='float32',
      attn_gate=False,
      attn_qk_norm=False,
    )
  )


def _make_gdn_config() -> SimpleNamespace:
  # seq_len must clear the 64-token chunk size: the tests run GDN in training
  # mode, which pins the `chunk` kernel.
  return _normalize_config(
    dict(
      arch_id='gdn',
      token_mixer='gdn',
      vocab_size=256,
      dim=64,
      n_heads=4,
      n_layers=1,
      seq_len=128,
      expand=4,
      mlp='glu',
      rmsnorm_eps=1e-6,
      model_dtype='float32',
      gdn_conv_size=4,
      gdn_gate=True,
      gdn_neg_eigval=True,
      intra_doc=False,
    )
  )


def _make_model_config(token_mixer: str, hybrid_mixer_ratio: int) -> SimpleNamespace:
  return _normalize_config(
    dict(
      token_mixer=token_mixer,
      hybrid_mixer_ratio=hybrid_mixer_ratio,
      vocab_size=256,
      dim=64,
      n_heads=4,
      n_layers=4,
      seq_len=128,
      expand=4,
      mlp='glu',
      rmsnorm_eps=1e-6,
      model_dtype='float32',
      tie_embeddings=True,
      attn_gate=True,
      attn_qk_norm=True,
      gdn_conv_size=4,
      gdn_gate=True,
      gdn_neg_eigval=True,
      intra_doc=False,
    )
  )


def _clone_state_dict(state_dict: dict) -> dict:
  return {key: tensor.detach().clone().to(DEVICE) for key, tensor in state_dict.items()}


def _to_device(state_dict: dict) -> dict:
  return {key: tensor.to(DEVICE) for key, tensor in state_dict.items()}


def _set_module_mode(module: nn.Module, module_spec: str) -> None:
  """Put a module in the mode that selects the kernels pretraining uses.

  `GatedDeltaNet.forward` routes to `fused_recurrent` whenever q_len <= 64
  outside training. That kernel is inference only: its backward raises
  NotImplementedError. Training mode pins the `chunk` kernel for both the
  forward and the backward, which is what pretraining runs. Neither backend has
  dropout, so training mode changes nothing else here.
  """
  if module_spec == 'gdn':
    module.train()
  else:
    module.eval()


def _gradients(module: nn.Module) -> dict:
  return {name: parameter.grad for name, parameter in module.named_parameters() if parameter.grad is not None}


def _init_weights(module: nn.Module) -> None:
  if isinstance(module, nn.Linear):
    torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    if module.bias is not None:
      torch.nn.init.zeros_(module.bias)
  elif isinstance(module, nn.Embedding):
    torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


RESIDUAL_BRANCH_SUFFIXES = (
  'fc2.weight',
  'w_out.weight',
  'o_proj.weight',
  'down_proj.weight',
)


def _scale_residual_branches(module: nn.Module, n_layers: int) -> None:
  """Scale the residual output projections of either backend.

  `Transformer._scale_residual_branches` lists the legacy names only. FLA names
  its attention output `o_proj.weight`, which the legacy list already covers for
  the legacy GDN, but names its FFN output `down_proj.weight`, which the legacy
  list does not cover. That name is included here so the same scaling reaches
  both backends.
  """
  for name, parameter in module.named_parameters():
    if name.endswith(RESIDUAL_BRANCH_SUFFIXES):
      torch.nn.init.normal_(parameter, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))


def _initialize(module: nn.Module, n_layers: int) -> None:
  """Initialize a module as `Transformer.__init__` initializes its submodules.

  `normal_` fills a tensor element by element, so drawing one fused legacy
  weight and drawing the separate FLA weights it splits into consume the same
  random numbers in the same order. Legacy and FLA also register those weights
  in the same order, so applying this to both backends under one seed gives
  them equal weights.
  """
  module.apply(_init_weights)
  _scale_residual_branches(module, n_layers)


def test_translated_module_fwd_pass(
  module_spec: str,
  legacy_state_dict: dict,
  legacy_config: dict | SimpleNamespace,
) -> None:
  if module_spec not in LEGACY_MODULE_SPEC_MAP:
    raise ValueError(f'Unknown module_spec {module_spec!r}; expected one of {sorted(LEGACY_MODULE_SPEC_MAP)}.')

  reference_tensor = next(iter(legacy_state_dict.values()))
  config = _normalize_config(legacy_config)
  fla_config = builder.config_builder(config)

  legacy_module = LEGACY_MODULE_SPEC_MAP[module_spec](config, fla_config)
  fla_module = FLA_MODULE_SPEC_MAP[module_spec](config, fla_config)

  legacy_module.to(device=DEVICE, dtype=reference_tensor.dtype)
  legacy_module.load_state_dict(_to_device(legacy_state_dict), strict=True)
  _set_module_mode(legacy_module, module_spec)

  fla_module.to(device=DEVICE, dtype=reference_tensor.dtype)
  translate_module = TRANSLATE_MODULE_SPEC_MAP[module_spec]
  translated_state_dict = translate_module(_to_device(legacy_state_dict))
  fla_module.load_state_dict(translated_state_dict, strict=True)
  _set_module_mode(fla_module, module_spec)

  legacy_module.to(dtype=torch.bfloat16, device=DEVICE)
  fla_module.to(dtype=torch.bfloat16, device=DEVICE)

  torch.manual_seed(0)
  inputs = _make_inputs(module_spec, config, legacy_module)

  with torch.no_grad():
    legacy_outputs = _forward(module_spec, legacy_module, inputs, config, fla_backend=False)
    fla_outputs = _forward(module_spec, fla_module, inputs, config, fla_backend=True)

  _print_metrics(_hidden_state_metrics(fla_outputs, legacy_outputs))


def test_initialization_fwd_pass(
  module_spec: str,
  legacy_config: dict | SimpleNamespace,
  seed: int = 0,
) -> None:
  """Initialize the legacy module and translate that initialization into FLA."""
  if module_spec not in LEGACY_MODULE_SPEC_MAP:
    raise ValueError(f'Unknown module_spec {module_spec!r}; expected one of {sorted(LEGACY_MODULE_SPEC_MAP)}.')

  config = _normalize_config(legacy_config)
  fla_config = builder.config_builder(config)
  dtype = getattr(torch, config.model_dtype)

  torch.manual_seed(seed)
  legacy_module = LEGACY_MODULE_SPEC_MAP[module_spec](config, fla_config).to(device=DEVICE, dtype=dtype)
  _initialize(legacy_module, config.n_layers)
  _set_module_mode(legacy_module, module_spec)

  fla_module = FLA_MODULE_SPEC_MAP[module_spec](config, fla_config).to(device=DEVICE, dtype=dtype)
  translate_module = TRANSLATE_MODULE_SPEC_MAP[module_spec]
  fla_module.load_state_dict(translate_module(legacy_module.state_dict()), strict=True)
  _set_module_mode(fla_module, module_spec)

  legacy_module.to(dtype=torch.bfloat16, device=DEVICE)
  fla_module.to(dtype=torch.bfloat16, device=DEVICE)

  torch.manual_seed(seed)
  inputs = _make_inputs(module_spec, config, legacy_module)

  with torch.no_grad():
    legacy_outputs = _forward(module_spec, legacy_module, inputs, config, fla_backend=False)
    fla_outputs = _forward(module_spec, fla_module, inputs, config, fla_backend=True)

  _print_metrics(_hidden_state_metrics(fla_outputs, legacy_outputs))
  print("\n")


def test_seeded_initialization_fwd_pass(
  module_spec: str,
  legacy_config: dict | SimpleNamespace,
  seed: int = 0,
) -> None:
  """Initialize both backends independently under one seed, without translation."""
  if module_spec not in LEGACY_MODULE_SPEC_MAP:
    raise ValueError(f'Unknown module_spec {module_spec!r}; expected one of {sorted(LEGACY_MODULE_SPEC_MAP)}.')

  config = _normalize_config(legacy_config)
  fla_config = builder.config_builder(config)
  dtype = getattr(torch, config.model_dtype)

  torch.manual_seed(seed)
  legacy_module = LEGACY_MODULE_SPEC_MAP[module_spec](config, fla_config).to(device=DEVICE, dtype=dtype)
  _initialize(legacy_module, config.n_layers)
  _set_module_mode(legacy_module, module_spec)

  torch.manual_seed(seed)
  fla_module = FLA_MODULE_SPEC_MAP[module_spec](config, fla_config).to(device=DEVICE, dtype=dtype)
  _initialize(fla_module, config.n_layers)
  _set_module_mode(fla_module, module_spec)

  translate_module = TRANSLATE_MODULE_SPEC_MAP[module_spec]
  expected_state_dict = translate_module(legacy_module.state_dict())
  fla_state_dict = fla_module.state_dict()

  for name in sorted(expected_state_dict):
    print(f'{name}:')
    _print_metrics(_tensor_metrics(fla_state_dict[name], expected_state_dict[name]), indent='  ')

  legacy_module.to(dtype=torch.bfloat16, device=DEVICE)
  fla_module.to(dtype=torch.bfloat16, device=DEVICE)

  torch.manual_seed(seed)
  inputs = _make_inputs(module_spec, config, legacy_module)

  with torch.no_grad():
    legacy_outputs = _forward(module_spec, legacy_module, inputs, config, fla_backend=False)
    fla_outputs = _forward(module_spec, fla_module, inputs, config, fla_backend=True)

  _print_metrics(_hidden_state_metrics(fla_outputs, legacy_outputs))
  print("\n")


def test_translated_module_bwd_pass(
  module_spec: str,
  legacy_state_dict: dict,
  legacy_config: dict | SimpleNamespace,
) -> None:
  if module_spec not in LEGACY_MODULE_SPEC_MAP:
    raise ValueError(f'Unknown module_spec {module_spec!r}; expected one of {sorted(LEGACY_MODULE_SPEC_MAP)}.')

  reference_tensor = next(iter(legacy_state_dict.values()))
  config = _normalize_config(legacy_config)
  fla_config = builder.config_builder(config)

  legacy_module = LEGACY_MODULE_SPEC_MAP[module_spec](config, fla_config)
  fla_module = FLA_MODULE_SPEC_MAP[module_spec](config, fla_config)

  legacy_module.to(device=DEVICE, dtype=reference_tensor.dtype)
  legacy_module.load_state_dict(_clone_state_dict(legacy_state_dict), strict=True)
  _set_module_mode(legacy_module, module_spec)
  legacy_module.zero_grad(set_to_none=True)

  fla_module.to(device=DEVICE, dtype=reference_tensor.dtype)
  translate_module = TRANSLATE_MODULE_SPEC_MAP[module_spec]
  translated_state_dict = translate_module(_clone_state_dict(legacy_state_dict))
  fla_module.load_state_dict(_clone_state_dict(translated_state_dict), strict=True)
  _set_module_mode(fla_module, module_spec)
  fla_module.zero_grad(set_to_none=True)

  legacy_module.to(dtype=torch.bfloat16, device=DEVICE)
  fla_module.to(dtype=torch.bfloat16, device=DEVICE)

  torch.manual_seed(0)
  inputs = _make_inputs(module_spec, config, legacy_module)

  legacy_outputs = _forward(module_spec, legacy_module, inputs, config, fla_backend=False)
  legacy_outputs.float().sum().backward()

  fla_outputs = _forward(module_spec, fla_module, inputs, config, fla_backend=True)
  fla_outputs.float().sum().backward()

  legacy_gradients = _gradients(legacy_module)
  if module_spec != 'gdn':
    # GDN translation is the identity, so the names already match. Running the
    # strict translator on a gradient dict would reject it for the parameters
    # `_gradients` drops.
    legacy_gradients = translate_module(legacy_gradients)
  fla_gradients = _gradients(fla_module)

  for name in sorted(legacy_gradients):
    if name not in fla_gradients:
      print(f'{name}: missing FLA gradient')
      continue

    print(f'{name}:')
    _print_metrics(_tensor_metrics(fla_gradients[name], legacy_gradients[name]), indent='  ')

  for name in sorted(fla_gradients.keys() - legacy_gradients.keys()):
    print(f'{name}: no legacy counterpart')
    print("\n")


def _legacy_model_config(config: SimpleNamespace) -> ModelConfig:
  return ModelConfig(
    vocab_size=config.vocab_size,
    seq_len=config.seq_len,
    dim=config.dim,
    expand=float(Fraction(config.expand)),
    n_layers=config.n_layers,
    n_heads=config.n_heads,
    mlp=config.mlp,
    rmsnorm_eps=config.rmsnorm_eps,
    tie_embeddings=getattr(config, 'tie_embeddings', False),
    model_dtype=config.model_dtype,
    token_mixer=getattr(config, 'token_mixer', 'attn'),
    hybrid_mixer_ratio=getattr(config, 'hybrid_mixer_ratio', 1),
    attn_gate=config.attn_gate,
    attn_qk_norm=config.attn_qk_norm,
    gdn_conv_size=config.gdn_conv_size,
    gdn_gate=config.gdn_gate,
    gdn_neg_eigval=config.gdn_neg_eigval,
    intra_doc=config.intra_doc,
  )


def test_translated_model_fwd_pass(legacy_config: dict | SimpleNamespace, seed: int = 0) -> None:
  """Translate a whole legacy Transformer into the FLA model and compare logits."""
  config = _normalize_config(legacy_config)
  fla_config = builder.config_builder(config)
  _, ratio = builder.parse_arch_id(config.arch_id)
  if config.arch_id == 'attn':
    attn_layers = list(range(config.n_layers))
  elif ratio is None:
    attn_layers = []
  else:
    attn_layers = builder.build_hybrid_layers(config.n_layers, ratio)
  print(f'arch_id: {config.arch_id}, attention layers: {attn_layers}')

  torch.manual_seed(seed)
  legacy_model = Transformer(_legacy_model_config(config)).to(device=DEVICE, dtype=torch.float32)
  legacy_model.train()

  fla_model = AutoModelForCausalLM.from_config(fla_config).to(device=DEVICE, dtype=torch.float32)
  fla_model.load_state_dict(
    translate_model(
      legacy_model.state_dict(),
      attn_layers=attn_layers,
      n_layers=config.n_layers,
      num_heads=config.n_heads,
    ),
    strict=True,
  )
  fla_model.train()

  legacy_model.to(dtype=torch.bfloat16, device=DEVICE)
  fla_model.to(dtype=torch.bfloat16, device=DEVICE)

  torch.manual_seed(seed)
  inputs = torch.randint(low=0, high=config.vocab_size, size=(2, config.seq_len), device=DEVICE)

  with torch.no_grad():
    legacy_outputs = legacy_model(inputs)
    fla_outputs = fla_model(inputs).logits

  _print_metrics(_hidden_state_metrics(fla_outputs, legacy_outputs))


def _run_module_tests(module_spec: str, config: SimpleNamespace) -> None:
  fla_config = builder.config_builder(config)

  torch.manual_seed(0)
  legacy_module = LEGACY_MODULE_SPEC_MAP[module_spec](config, fla_config).to(device=DEVICE, dtype=torch.float32)
  legacy_state_dict = legacy_module.state_dict()

  print(f'--- {module_spec} forward ---')
  test_translated_module_fwd_pass(module_spec, legacy_state_dict, config)

  print(f'--- {module_spec} backward ---')
  test_translated_module_bwd_pass(module_spec, legacy_state_dict, config)

  print(f'--- {module_spec} initialization forward (translated) ---')
  test_initialization_fwd_pass(module_spec, config)

  print(f'--- {module_spec} initialization forward (seeded) ---')
  test_seeded_initialization_fwd_pass(module_spec, config)


def main() -> None:
  config = _make_legacy_config()

  for module_spec in LEGACY_MODULE_SPEC_MAP:
    if module_spec == 'gdn':
      continue
    _run_module_tests(module_spec, config)

  _run_module_tests('gdn', _make_gdn_config())

  for token_mixer, hybrid_mixer_ratio in (('attn', 1), ('gdn', 1), ('gdn+attn', 3), ('gdn+attn', -3)):
    model_config = _make_model_config(token_mixer, hybrid_mixer_ratio)
    print(f'=== model {model_config.arch_id} ===')
    test_translated_model_fwd_pass(model_config)


if __name__ == '__main__':
  main()

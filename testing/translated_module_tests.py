from __future__ import annotations

import math
from collections.abc import Callable
from fractions import Fraction
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from fla.layers.attn import Attention
from fla.modules import GatedMLP
from fla.modules import RMSNorm as FLARMSNorm
from src.models import builder
from src.models.legacy.attention import GatedAttention
from src.models.legacy.components import GLU, RMSNorm
from src.models.legacy.embeddings import precompute_freqs_cis
from src.models.translate_backend import (
  translate_embeddings,
  translate_ffn,
  translate_gated_attention,
  translate_lm_head,
  translate_rms_norm,
)

ModuleFactory = Callable[[SimpleNamespace, Any], nn.Module]


def _legacy_gated_attention(config: SimpleNamespace, fla_config: Any) -> nn.Module:
  return GatedAttention(config)


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
  'ffn': _legacy_ffn,
  'embeddings': _legacy_embeddings,
  'rms_norm': _legacy_rms_norm,
  'lm_head': _legacy_lm_head,
}

FLA_MODULE_SPEC_MAP: dict[str, ModuleFactory] = {
  'gated_attention': _fla_gated_attention,
  'ffn': _fla_ffn,
  'embeddings': _fla_embeddings,
  'rms_norm': _fla_rms_norm,
  'lm_head': _fla_lm_head,
}

TRANSLATE_MODULE_SPEC_MAP: dict[str, Callable[[dict], dict]] = {
  'gated_attention': translate_gated_attention,
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
  return SimpleNamespace(**values)


def _make_inputs(module_spec: str, config: SimpleNamespace, module: nn.Module) -> torch.Tensor:
  parameter = next(module.parameters())
  batch_size = 2
  sequence_length = min(config.seq_len, 16)
  if module_spec == 'embeddings':
    return torch.randint(
      low=0,
      high=config.vocab_size,
      size=(batch_size, sequence_length),
      device=parameter.device,
    )
  return torch.randn(
    batch_size,
    sequence_length,
    config.dim,
    device=parameter.device,
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
  if module_spec != 'gated_attention':
    return module(inputs)

  if fla_backend:
    return module(inputs, permute_rope_qk=True)[0]

  head_dim = config.dim // config.n_heads
  freqs_cis = precompute_freqs_cis(head_dim, inputs.shape[1], theta=500000).to(inputs.device)
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


def _clone_state_dict(state_dict: dict) -> dict:
  return {key: tensor.detach().clone() for key, tensor in state_dict.items()}


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

  legacy_module.to(device=reference_tensor.device, dtype=reference_tensor.dtype)
  legacy_module.load_state_dict(legacy_state_dict, strict=True)
  legacy_module.eval()

  fla_module.to(device=reference_tensor.device, dtype=reference_tensor.dtype)
  translate_module = TRANSLATE_MODULE_SPEC_MAP[module_spec]
  translated_state_dict = translate_module(legacy_state_dict)
  fla_module.load_state_dict(translated_state_dict, strict=True)
  fla_module.eval()

  torch.manual_seed(0)
  inputs = _make_inputs(module_spec, config, legacy_module)

  with torch.no_grad():
    legacy_outputs = _forward(module_spec, legacy_module, inputs, config, fla_backend=False)
    fla_outputs = _forward(module_spec, fla_module, inputs, config, fla_backend=True)

  difference = fla_outputs.float() - legacy_outputs.float()

  print(f'allclose: {torch.allclose(fla_outputs, legacy_outputs)}')
  print(f'frobenius norm: {torch.linalg.vector_norm(difference)}')
  print(f'max absolute error: {difference.abs().max()}')


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
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  dtype = getattr(torch, config.model_dtype)

  torch.manual_seed(seed)
  legacy_module = LEGACY_MODULE_SPEC_MAP[module_spec](config, fla_config).to(device=device, dtype=dtype)
  _initialize(legacy_module, config.n_layers)
  legacy_module.eval()

  fla_module = FLA_MODULE_SPEC_MAP[module_spec](config, fla_config).to(device=device, dtype=dtype)
  translate_module = TRANSLATE_MODULE_SPEC_MAP[module_spec]
  fla_module.load_state_dict(translate_module(legacy_module.state_dict()), strict=True)
  fla_module.eval()

  torch.manual_seed(seed)
  inputs = _make_inputs(module_spec, config, legacy_module)

  with torch.no_grad():
    legacy_outputs = _forward(module_spec, legacy_module, inputs, config, fla_backend=False)
    fla_outputs = _forward(module_spec, fla_module, inputs, config, fla_backend=True)

  difference = fla_outputs.float() - legacy_outputs.float()

  print(f'allclose: {torch.allclose(fla_outputs, legacy_outputs)}')
  print(f'frobenius norm: {torch.linalg.vector_norm(difference)}')
  print(f'max absolute error: {difference.abs().max()}')


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
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  dtype = getattr(torch, config.model_dtype)

  torch.manual_seed(seed)
  legacy_module = LEGACY_MODULE_SPEC_MAP[module_spec](config, fla_config).to(device=device, dtype=dtype)
  _initialize(legacy_module, config.n_layers)
  legacy_module.eval()

  torch.manual_seed(seed)
  fla_module = FLA_MODULE_SPEC_MAP[module_spec](config, fla_config).to(device=device, dtype=dtype)
  _initialize(fla_module, config.n_layers)
  fla_module.eval()

  translate_module = TRANSLATE_MODULE_SPEC_MAP[module_spec]
  expected_state_dict = translate_module(legacy_module.state_dict())
  fla_state_dict = fla_module.state_dict()

  for name in sorted(expected_state_dict):
    expected = expected_state_dict[name].float()
    actual = fla_state_dict[name].float()
    print(f'{name}: max absolute error {(actual - expected).abs().max()}')

  torch.manual_seed(seed)
  inputs = _make_inputs(module_spec, config, legacy_module)

  with torch.no_grad():
    legacy_outputs = _forward(module_spec, legacy_module, inputs, config, fla_backend=False)
    fla_outputs = _forward(module_spec, fla_module, inputs, config, fla_backend=True)

  difference = fla_outputs.float() - legacy_outputs.float()

  print(f'allclose: {torch.allclose(fla_outputs, legacy_outputs)}')
  print(f'frobenius norm: {torch.linalg.vector_norm(difference)}')
  print(f'max absolute error: {difference.abs().max()}')


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

  legacy_module.to(device=reference_tensor.device, dtype=reference_tensor.dtype)
  legacy_module.load_state_dict(_clone_state_dict(legacy_state_dict), strict=True)
  legacy_module.eval()
  legacy_module.zero_grad(set_to_none=True)

  fla_module.to(device=reference_tensor.device, dtype=reference_tensor.dtype)
  translate_module = TRANSLATE_MODULE_SPEC_MAP[module_spec]
  translated_state_dict = translate_module(_clone_state_dict(legacy_state_dict))
  fla_module.load_state_dict(_clone_state_dict(translated_state_dict), strict=True)
  fla_module.eval()
  fla_module.zero_grad(set_to_none=True)

  torch.manual_seed(0)
  inputs = _make_inputs(module_spec, config, legacy_module)

  legacy_outputs = _forward(module_spec, legacy_module, inputs, config, fla_backend=False)
  legacy_outputs.float().sum().backward()

  fla_outputs = _forward(module_spec, fla_module, inputs, config, fla_backend=True)
  fla_outputs.float().sum().backward()

  legacy_gradients = translate_module(_gradients(legacy_module))
  fla_gradients = _gradients(fla_module)

  for name in sorted(legacy_gradients):
    if name not in fla_gradients:
      print(f'{name}: missing FLA gradient')
      continue

    legacy_gradient = legacy_gradients[name]
    fla_gradient = fla_gradients[name]
    difference = fla_gradient.float() - legacy_gradient.float()

    print(f'{name}:')
    print(f'  equal: {torch.equal(fla_gradient, legacy_gradient)}')
    print(f'  allclose: {torch.allclose(fla_gradient, legacy_gradient)}')
    print(f'  frobenius norm: {torch.linalg.vector_norm(difference)}')

  for name in sorted(fla_gradients.keys() - legacy_gradients.keys()):
    print(f'{name}: no legacy counterpart')


def main() -> None:
  config = _make_legacy_config()
  fla_config = builder.config_builder(config)
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  dtype = torch.float32

  for module_spec, factory in LEGACY_MODULE_SPEC_MAP.items():
    torch.manual_seed(0)
    legacy_module = factory(config, fla_config).to(device=device, dtype=dtype)
    legacy_state_dict = legacy_module.state_dict()

    print(f'--- {module_spec} forward ---')
    test_translated_module_fwd_pass(module_spec, legacy_state_dict, config)

    print(f'--- {module_spec} backward ---')
    test_translated_module_bwd_pass(module_spec, legacy_state_dict, config)

    print(f'--- {module_spec} initialization forward (translated) ---')
    test_initialization_fwd_pass(module_spec, config)

    print(f'--- {module_spec} initialization forward (seeded) ---')
    test_seeded_initialization_fwd_pass(module_spec, config)


if __name__ == '__main__':
  main()

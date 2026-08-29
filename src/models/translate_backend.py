from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from torch import Tensor


def _translate_weight_only_state_dict(
  state_dict: Mapping[str, Tensor],
  *,
  layer_name: str,
  expected_ndim: int,
) -> dict[str, Tensor]:
  required_keys = {'weight'}
  missing_keys = required_keys - state_dict.keys()
  unexpected_keys = state_dict.keys() - required_keys
  if missing_keys or unexpected_keys:
    raise ValueError(
      f'Invalid legacy {layer_name} state dict: missing keys={sorted(missing_keys)}, '
      f'unexpected keys={sorted(unexpected_keys)}'
    )

  weight = state_dict['weight']
  if weight.ndim != expected_ndim:
    raise ValueError(
      f'Expected legacy {layer_name} weight to be {expected_ndim}-dimensional, got shape {tuple(weight.shape)}.'
    )
  return {'weight': weight}


def translate_embeddings(state_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
  """Translate one legacy token embedding state dict to FLA embedding keys.

  Pass layer-local keys, without a model prefix. The target must use the same
  vocabulary and hidden sizes. The input is not modified, and the translated
  weight retains its dtype, device, and storage.
  """
  return _translate_weight_only_state_dict(state_dict, layer_name='embedding', expected_ndim=2)


def translate_rms_norm(state_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
  """Translate one legacy RMSNorm state dict to FLA RMSNorm keys.

  Pass layer-local keys, without a model/layer prefix. The target must use the
  same hidden size and epsilon, with elementwise_affine=True and bias=False.
  The input is not modified, and the translated weight retains its dtype,
  device, and storage.
  """
  return _translate_weight_only_state_dict(state_dict, layer_name='RMSNorm', expected_ndim=1)


def translate_lm_head(state_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
  """Translate one legacy bias-free LM head state dict to FLA LM head keys.

  Pass layer-local keys, without a model prefix. The target must use the same
  hidden and vocabulary sizes and bias=False. The input is not modified, and
  the translated weight retains its dtype, device, and storage.
  """
  return _translate_weight_only_state_dict(state_dict, layer_name='LM head', expected_ndim=2)


def translate_ffn(state_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
  """Translate one legacy GLU state dict to FLA GatedMLP keys.

  Pass layer-local keys, without a model/layer prefix. The target must use the
  same hidden and intermediate sizes as the legacy layer. The input is not
  modified, and the translated tensors retain their dtype and device while
  sharing storage with the input tensors.
  """
  required_keys = {'fc1.weight', 'fc2.weight'}
  missing_keys = required_keys - state_dict.keys()
  unexpected_keys = state_dict.keys() - required_keys
  if missing_keys or unexpected_keys:
    raise ValueError(
      f'Invalid legacy FFN state dict: missing keys={sorted(missing_keys)}, '
      f'unexpected keys={sorted(unexpected_keys)}'
    )

  fc1_weight = state_dict['fc1.weight']
  fc2_weight = state_dict['fc2.weight']
  if fc1_weight.ndim != 2:
    raise ValueError(f'Expected fc1.weight to be two-dimensional, got shape {tuple(fc1_weight.shape)}.')
  if fc2_weight.ndim != 2:
    raise ValueError(f'Expected fc2.weight to be two-dimensional, got shape {tuple(fc2_weight.shape)}.')

  hidden_size, intermediate_size = fc2_weight.shape
  expected_fc1_shape = (2 * intermediate_size, hidden_size)
  if fc1_weight.shape != expected_fc1_shape:
    raise ValueError(
      'Legacy and FLA FFNs are equivalent only for a gated legacy GLU: '
      f'expected fc1.weight with shape {expected_fc1_shape} based on fc2.weight, '
      f'got {tuple(fc1_weight.shape)}.'
    )

  gate_weight, up_weight = fc1_weight.split(intermediate_size, dim=0)
  return {
    'gate_proj.weight': gate_weight,
    'up_proj.weight': up_weight,
    'down_proj.weight': fc2_weight,
  }


def translate_gated_attention(
  state_dict: Mapping[str, Tensor],
  *,
  permute_rope_weights: bool = False,
  num_heads: int | None = None,
) -> dict[str, Tensor]:
  """Translate one legacy GatedAttention state dict to FLA Attention keys.

  Pass layer-local keys, without a model/layer prefix. The target must use the
  same hidden size and head count, num_kv_heads=num_heads, qkv_bias=False, and
  matching use_gate/qk_norm settings. Normalization epsilon and attention
  masking must be matched separately; neither is encoded in the state dict.

  With permute_rope_weights=True, num_heads is required. Q/K projection rows
  and norm weights are reordered within each head from adjacent pairs to
  even coordinates followed by odd coordinates, matching FLA's non-interleaved
  RoPE. Leave FLA's forward permute_rope_qk=False in this case. Alternatively,
  leave weights unpermuted and call FLA with permute_rope_qk=True. Enable only
  one permutation path; both still apply RoPE. Set rope_theta=500000 to match
  the legacy Transformer and window_size=None for full attention.

  The input is not modified. Existing weights retain their dtype and device
  and may share storage with the input. Permuting can allocate new weights;
  the zero gate bias is always newly allocated when gating is enabled.
  Remaining shape checks are delegated to target.load_state_dict(..., strict=True).
  """
  required_keys = {'w_qkv.weight', 'w_out.weight'}
  norm_keys = {'q_norm.weight', 'k_norm.weight'}
  allowed_keys = required_keys | norm_keys | {'w_gate.weight'}
  missing_keys = required_keys - state_dict.keys()
  unexpected_keys = state_dict.keys() - allowed_keys
  if missing_keys or unexpected_keys:
    raise ValueError(
      f'Invalid legacy attention state dict: missing keys={sorted(missing_keys)}, '
      f'unexpected keys={sorted(unexpected_keys)}'
    )
  if norm_keys & state_dict.keys() and not norm_keys <= state_dict.keys():
    raise ValueError('Legacy QK normalization requires both q_norm.weight and k_norm.weight.')

  qkv_weight = state_dict['w_qkv.weight']
  if qkv_weight.ndim != 2 or qkv_weight.shape[1] == 0 or qkv_weight.shape[0] != 3 * qkv_weight.shape[1]:
    raise ValueError(f'Expected w_qkv.weight with shape (3 * dim, dim), got {tuple(qkv_weight.shape)}.')
  hidden_size = qkv_weight.shape[1]
  q_weight, k_weight, v_weight = qkv_weight.split(hidden_size, dim=0)
  translated = {
    'q_proj.weight': q_weight,
    'k_proj.weight': k_weight,
    'v_proj.weight': v_weight,
    'o_proj.weight': state_dict['w_out.weight'],
  }

  if 'w_gate.weight' in state_dict:
    gate_weight = state_dict['w_gate.weight']
    translated['gate.weight'] = gate_weight
    translated['gate.bias'] = gate_weight.new_zeros(hidden_size)

  if norm_keys <= state_dict.keys():
    translated['q_norm.weight'] = state_dict['q_norm.weight']
    translated['k_norm.weight'] = state_dict['k_norm.weight']

  if permute_rope_weights:
    if not isinstance(num_heads, int) or isinstance(num_heads, bool) or num_heads <= 0:
      raise ValueError('permute_rope_weights=True requires a positive integer num_heads.')
    if hidden_size % num_heads != 0:
      raise ValueError(f'Hidden size {hidden_size} must be divisible by num_heads={num_heads}.')
    head_dim = hidden_size // num_heads
    if head_dim % 2 != 0:
      raise ValueError(f'RoPE permutation requires an even head dimension, got {head_dim}.')

    for key in ('q_proj.weight', 'k_proj.weight'):
      translated[key] = (
        translated[key]
        .reshape(num_heads, head_dim // 2, 2, hidden_size)
        .transpose(1, 2)
        .reshape(hidden_size, hidden_size)
      )
    for key in ('q_norm.weight', 'k_norm.weight'):
      if key in translated:
        weight = translated[key]
        if weight.shape != (head_dim,):
          raise ValueError(f'Expected {key} with shape ({head_dim},), got {tuple(weight.shape)}.')
        translated[key] = weight.reshape(head_dim // 2, 2).transpose(0, 1).reshape(head_dim)

  return translated

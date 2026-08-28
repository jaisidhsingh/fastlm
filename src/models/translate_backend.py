from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from torch import Tensor


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

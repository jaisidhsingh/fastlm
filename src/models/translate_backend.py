from __future__ import annotations

from collections.abc import Collection, Mapping
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


def translate_gated_deltanet(
  state_dict: Mapping[str, Tensor],
  *,
  num_heads: int | None = None,
) -> dict[str, Tensor]:
  """Translate one legacy GatedDeltaNet state dict to FLA GatedDeltaNet keys.

  Pass layer-local keys, without a model/layer prefix. The mapping is the
  identity: `fla.layers.legacy_gated_deltanet.LegacyGatedDeltaNet` and
  `fla.layers.gated_deltanet.GatedDeltaNet` register the same parameters under
  the same names. Shapes are checked against each other, so a state dict that
  passes here is internally consistent but is not thereby matched to any target.

  The target must be constructed with the same expand_v, allow_neg_eigval,
  conv_size, use_gate, norm_eps, and intra_doc. None of these is encoded in the
  state dict. allow_neg_eigval in particular scales beta inside the kernel, so
  it changes the output without changing any shape.

  Pass num_heads to also check that the key dimension divides by the head count;
  the head count is not recoverable from the state dict, because a_proj and
  b_proj are sized by num_v_heads.

  The input is not modified. The returned dict is new, and its tensors are the
  input tensors, retaining their dtype and device and sharing storage.
  """
  required_keys = {
    'q_proj.weight',
    'k_proj.weight',
    'v_proj.weight',
    'a_proj.weight',
    'b_proj.weight',
    'A_log',
    'dt_bias',
    'o_norm.weight',
    'o_proj.weight',
  }
  conv_keys = {'q_conv1d.weight', 'k_conv1d.weight', 'v_conv1d.weight'}
  conv_bias_keys = {'q_conv1d.bias', 'k_conv1d.bias', 'v_conv1d.bias'}
  allowed_keys = required_keys | conv_keys | conv_bias_keys | {'g_proj.weight'}
  missing_keys = required_keys - state_dict.keys()
  unexpected_keys = state_dict.keys() - allowed_keys
  if missing_keys or unexpected_keys:
    raise ValueError(
      f'Invalid legacy GatedDeltaNet state dict: missing keys={sorted(missing_keys)}, '
      f'unexpected keys={sorted(unexpected_keys)}'
    )
  if conv_keys & state_dict.keys() and not conv_keys <= state_dict.keys():
    raise ValueError(f'Short convolutions require all of {sorted(conv_keys)}.')
  if conv_bias_keys & state_dict.keys():
    if not conv_bias_keys <= state_dict.keys():
      raise ValueError(f'Short convolution biases require all of {sorted(conv_bias_keys)}.')
    if not conv_keys <= state_dict.keys():
      raise ValueError(f'Short convolution biases require the weights {sorted(conv_keys)}.')

  q_weight = state_dict['q_proj.weight']
  if q_weight.ndim != 2 or q_weight.shape[1] == 0:
    raise ValueError(f'Expected q_proj.weight with shape (key_dim, hidden_size), got {tuple(q_weight.shape)}.')
  key_dim, hidden_size = q_weight.shape
  if state_dict['k_proj.weight'].shape != q_weight.shape:
    raise ValueError(
      f'Expected k_proj.weight with shape {tuple(q_weight.shape)} to match q_proj.weight, '
      f'got {tuple(state_dict["k_proj.weight"].shape)}.'
    )

  v_weight = state_dict['v_proj.weight']
  if v_weight.ndim != 2 or v_weight.shape[1] != hidden_size:
    raise ValueError(f'Expected v_proj.weight with shape (value_dim, {hidden_size}), got {tuple(v_weight.shape)}.')
  value_dim = v_weight.shape[0]

  a_weight = state_dict['a_proj.weight']
  if a_weight.ndim != 2 or a_weight.shape[1] != hidden_size:
    raise ValueError(f'Expected a_proj.weight with shape (num_v_heads, {hidden_size}), got {tuple(a_weight.shape)}.')
  num_v_heads = a_weight.shape[0]
  if state_dict['b_proj.weight'].shape != a_weight.shape:
    raise ValueError(
      f'Expected b_proj.weight with shape {tuple(a_weight.shape)} to match a_proj.weight, '
      f'got {tuple(state_dict["b_proj.weight"].shape)}.'
    )

  for key in ('A_log', 'dt_bias'):
    weight = state_dict[key]
    if weight.shape != (num_v_heads,):
      raise ValueError(f'Expected {key} with shape ({num_v_heads},), got {tuple(weight.shape)}.')

  o_weight = state_dict['o_proj.weight']
  if o_weight.shape != (hidden_size, value_dim):
    raise ValueError(f'Expected o_proj.weight with shape ({hidden_size}, {value_dim}), got {tuple(o_weight.shape)}.')

  if value_dim % num_v_heads != 0:
    raise ValueError(f'Value dimension {value_dim} must be divisible by num_v_heads={num_v_heads}.')
  head_v_dim = value_dim // num_v_heads
  o_norm_weight = state_dict['o_norm.weight']
  if o_norm_weight.shape != (head_v_dim,):
    raise ValueError(f'Expected o_norm.weight with shape ({head_v_dim},), got {tuple(o_norm_weight.shape)}.')

  if conv_keys <= state_dict.keys():
    conv_size = state_dict['q_conv1d.weight'].shape[-1]
    for key, channels in (
      ('q_conv1d.weight', key_dim),
      ('k_conv1d.weight', key_dim),
      ('v_conv1d.weight', value_dim),
    ):
      weight = state_dict[key]
      if weight.shape != (channels, 1, conv_size):
        raise ValueError(f'Expected {key} with shape ({channels}, 1, {conv_size}), got {tuple(weight.shape)}.')
    for key, channels in (
      ('q_conv1d.bias', key_dim),
      ('k_conv1d.bias', key_dim),
      ('v_conv1d.bias', value_dim),
    ):
      if key in state_dict and state_dict[key].shape != (channels,):
        raise ValueError(f'Expected {key} with shape ({channels},), got {tuple(state_dict[key].shape)}.')

  if 'g_proj.weight' in state_dict:
    gate_weight = state_dict['g_proj.weight']
    if gate_weight.shape != (value_dim, hidden_size):
      raise ValueError(
        f'Expected g_proj.weight with shape ({value_dim}, {hidden_size}), got {tuple(gate_weight.shape)}.'
      )

  if num_heads is not None:
    if not isinstance(num_heads, int) or isinstance(num_heads, bool) or num_heads <= 0:
      raise ValueError('num_heads must be a positive integer.')
    if key_dim % num_heads != 0:
      raise ValueError(f'Key dimension {key_dim} must be divisible by num_heads={num_heads}.')

  return dict(state_dict)


def translate_model(
  state_dict: Mapping[str, Tensor],
  *,
  attn_layers: Collection[int],
  n_layers: int,
  num_heads: int | None = None,
) -> dict[str, Tensor]:
  """Translate a whole legacy Transformer state dict to FLA CausalLM keys.

  This is the one function here that takes prefixed keys. Pass the state dict of
  a legacy `Transformer`, with any checkpoint prefix already stripped.

  `attn_layers` lists the layer indices that hold a legacy GatedAttention; every
  other index holds a legacy GatedDeltaNet. Build it with
  `builder.build_hybrid_layers`. It cannot be recovered from the key names: FLA
  names the token mixer `attn` for both layer types.

  Attention layers are translated with permute_rope_weights=True, so the target
  must be called with permute_rope_qk left at False, which is what the FLA model
  forward does. `num_heads` is required whenever `attn_layers` is not empty.

  `lm_head.weight` is always emitted. Weight tying is a property of the target,
  set through tie_word_embeddings on its config.

  The input is not modified. The returned dict is new; its tensors retain their
  dtype and device, and share storage with the input except where
  `translate_gated_attention` allocates.
  """
  attn_layers = set(attn_layers)
  unknown_layers = attn_layers - set(range(n_layers))
  if unknown_layers:
    raise ValueError(f'attn_layers holds indices outside range(n_layers={n_layers}): {sorted(unknown_layers)}.')
  if attn_layers and num_heads is None:
    raise ValueError('num_heads is required to translate attention layers.')

  translated: dict[str, Tensor] = {}
  remaining = dict(state_dict)

  def take(prefix: str) -> dict[str, Tensor]:
    taken = {key[len(prefix) :]: remaining.pop(key) for key in list(remaining) if key.startswith(prefix)}
    if not taken:
      raise ValueError(f'Invalid legacy Transformer state dict: no keys under {prefix!r}.')
    return taken

  def put(prefix: str, layer_state_dict: Mapping[str, Tensor]) -> None:
    translated.update({prefix + key: tensor for key, tensor in layer_state_dict.items()})

  put('model.embeddings.', translate_embeddings(take('embed_tokens.')))

  for layer in range(n_layers):
    legacy_prefix = f'layers.{layer}.'
    fla_prefix = f'model.layers.{layer}.'
    put(fla_prefix + 'attn_norm.', translate_rms_norm(take(legacy_prefix + 'token_mixer_norm.')))
    put(fla_prefix + 'mlp_norm.', translate_rms_norm(take(legacy_prefix + 'mlp_norm.')))
    put(fla_prefix + 'mlp.', translate_ffn(take(legacy_prefix + 'mlp.')))

    token_mixer = take(legacy_prefix + 'token_mixer.')
    if layer in attn_layers:
      token_mixer = translate_gated_attention(token_mixer, permute_rope_weights=True, num_heads=num_heads)
    else:
      token_mixer = translate_gated_deltanet(token_mixer, num_heads=num_heads)
    put(fla_prefix + 'attn.', token_mixer)

  put('model.norm.', translate_rms_norm(take('out_norm.')))
  put('lm_head.', translate_lm_head(take('lm_head.')))

  if remaining:
    raise ValueError(f'Invalid legacy Transformer state dict: unexpected keys={sorted(remaining)}')
  return translated

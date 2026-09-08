from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING, Any

import torch

from src.models.legacy.construct import get_param_groups
from src.models.translate_backend import translate_model

if TYPE_CHECKING:
  from torch import Tensor, nn


ADAMW_STATE_KEYS = frozenset({'step', 'exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'})
ADAMW_MOMENT_KEYS = ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq')
REQUIRED_ADAMW_MOMENT_KEYS = frozenset({'exp_avg', 'exp_avg_sq'})


def ordered_param_names(model: nn.Module, weight_decay: float) -> list[list[str]]:
  """List the parameter names of each optimizer group, in optimizer index order.

  `get_param_groups` returns Parameter objects. This resolves each one back to
  its name by identity, so the ordering rule is not duplicated here and cannot
  drift from the one the engine uses.
  """
  name_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
  groups: list[list[str]] = []
  for group in get_param_groups(model, weight_decay):
    names = []
    for parameter in group['params']:
      name = name_by_id.get(id(parameter))
      if name is None:
        raise ValueError('A parameter returned by get_param_groups is not in model.named_parameters().')
      names.append(name)
    groups.append(names)
  return groups


def _flatten(groups: Collection[list[str]]) -> list[str]:
  return [name for group in groups for name in group]


def _group_index_by_name(groups: Collection[list[str]]) -> dict[str, int]:
  return {name: index for index, group in enumerate(groups) for name in group}


def add_tied_lm_head(named_tensors: dict[str, Tensor]) -> dict[str, Tensor]:
  """Add `lm_head.weight` when the legacy model ties it to the embedding.

  `named_parameters()` deduplicates a tied weight, so it reports one name.
  `translate_model` takes a `state_dict()`, which does not deduplicate, and
  requires both keys.
  """
  if 'lm_head.weight' not in named_tensors:
    if 'embed_tokens.weight' not in named_tensors:
      raise ValueError('Legacy parameter names hold neither lm_head.weight nor embed_tokens.weight.')
    named_tensors = dict(named_tensors)
    named_tensors['lm_head.weight'] = named_tensors['embed_tokens.weight']
  return named_tensors


def recover_name_map(
  legacy_model: nn.Module,
  legacy_names: Collection[str],
  *,
  attn_layers: Collection[int],
  n_layers: int,
  num_heads: int | None,
) -> dict[str, str | None]:
  """Map each FLA parameter name to the legacy parameter name that feeds it.

  The map is observed from `translate_model` rather than re-derived. Each legacy
  parameter is filled with a distinct constant `i + 1`. Every transform in
  `translate_model` is a split, reshape, transpose or `new_zeros`, so every
  output tensor is still constant valued and names its source. A value of `0`
  marks a tensor `translate_model` synthesized, which is `gate.bias`.
  """
  legacy_names = list(legacy_names)
  shape_by_name = {name: parameter.shape for name, parameter in legacy_model.named_parameters()}
  probe = {
    name: torch.full(shape_by_name[name], float(index + 1), dtype=torch.float64)
    for index, name in enumerate(legacy_names)
  }
  translated = translate_model(
    add_tied_lm_head(probe),
    attn_layers=attn_layers,
    n_layers=n_layers,
    num_heads=num_heads,
  )

  name_map: dict[str, str | None] = {}
  for fla_name, tensor in translated.items():
    minimum = tensor.min().item()
    if minimum != tensor.max().item():
      raise ValueError(
        f'translate_model mixes coordinates for {fla_name!r}: the probe tensor is not constant. '
        'Optimizer state cannot be translated by name alone.'
      )
    source_index = int(round(minimum)) - 1
    if source_index < 0:
      name_map[fla_name] = None
    else:
      name_map[fla_name] = legacy_names[source_index]
  return name_map


def _synthesized_source(fla_name: str, name_map: Mapping[str, str | None]) -> str:
  """Name the legacy parameter a synthesized FLA parameter belongs to.

  `translate_gated_attention` synthesizes only `gate.bias`, whose sibling
  `gate.weight` carries the legacy source.
  """
  if not fla_name.endswith('.bias'):
    raise ValueError(f'Unexpected synthesized FLA parameter {fla_name!r}; only a bias can be synthesized.')
  sibling = fla_name[: -len('.bias')] + '.weight'
  source = name_map.get(sibling)
  if source is None:
    raise ValueError(f'Synthesized FLA parameter {fla_name!r} has no source for its sibling {sibling!r}.')
  return source


def _check_adamw_state(state: Mapping[int, Mapping[str, Any]]) -> tuple[str, ...]:
  moment_keys: set[str] = set()
  for index, entry in state.items():
    unexpected = entry.keys() - ADAMW_STATE_KEYS
    if unexpected:
      raise ValueError(
        f'Optimizer state for parameter {index} holds keys {sorted(unexpected)}, which AdamW does not use. '
        'Only AdamW state can be translated.'
      )
    missing = REQUIRED_ADAMW_MOMENT_KEYS - entry.keys()
    if missing:
      raise ValueError(f'Optimizer state for parameter {index} is missing AdamW keys {sorted(missing)}.')
    moment_keys |= entry.keys() & set(ADAMW_MOMENT_KEYS)
  for key in moment_keys:
    for index, entry in state.items():
      if key not in entry:
        raise ValueError(f'Optimizer state for parameter {index} is missing {key!r}, which other parameters hold.')
  return tuple(key for key in ADAMW_MOMENT_KEYS if key in moment_keys)


def translate_adamw(
  optimizer_state_dict: Mapping[str, Any],
  *,
  legacy_model: nn.Module,
  fla_model: nn.Module,
  weight_decay: float,
  attn_layers: Collection[int],
  n_layers: int,
  num_heads: int | None = None,
) -> dict[str, Any]:
  """Translate a legacy AdamW `state_dict()` onto the FLA backend.

  Pass the state dict of an AdamW built on `get_param_groups(legacy_model,
  weight_decay)`, which is what `src/engine/engine.py` builds. `weight_decay`
  must be the value that run used, because it selects the same grouping on both
  models. `attn_layers`, `n_layers` and `num_heads` are the arguments
  `translate_model` takes; build `attn_layers` with `builder.build_hybrid_layers`.

  The returned dict is ready for `AdamW.load_state_dict` on an optimizer built as
  `intialize_optimizer(get_param_groups(fla_model, weight_decay), cfg)`.

  Moments are translated with `translate_model`, which is valid because AdamW is
  elementwise and every transform there is a reordering of independent
  coordinates. FLA's attention gate bias has no legacy source, so it takes zero
  moments and the `step` of the gate weight it accompanies.

  Only AdamW state is accepted. Dtype and device are left alone, because
  `AdamW.load_state_dict` casts state to each target parameter and places `step`
  according to the target's own `fused` and `capturable` settings.

  The returned moments share storage with `optimizer_state_dict`, and
  `AdamW.load_state_dict` keeps them whenever dtype and device already match.
  The target optimizer then writes into the source checkpoint on its first step.
  Clone the moments first if the source optimizer is still in use, which is the
  case when both backends run side by side in a test.
  """
  missing_top_level = {'state', 'param_groups'} - optimizer_state_dict.keys()
  if missing_top_level:
    raise ValueError(f'Optimizer state dict is missing keys {sorted(missing_top_level)}.')

  legacy_groups = ordered_param_names(legacy_model, weight_decay)
  fla_groups = ordered_param_names(fla_model, weight_decay)
  source_groups = optimizer_state_dict['param_groups']

  if len(source_groups) != len(legacy_groups):
    raise ValueError(
      f'Optimizer state dict holds {len(source_groups)} parameter groups, '
      f'but get_param_groups(legacy_model, weight_decay={weight_decay}) gives {len(legacy_groups)}.'
    )
  for group_index, (source_group, names) in enumerate(zip(source_groups, legacy_groups, strict=True)):
    if len(source_group['params']) != len(names):
      raise ValueError(
        f'Optimizer parameter group {group_index} holds {len(source_group["params"])} parameters, '
        f'but the legacy model gives {len(names)}. The checkpoint was written by a different model.'
      )

  legacy_names = _flatten(legacy_groups)
  index_by_legacy_name = {}
  position = 0
  for source_group in source_groups:
    for index in source_group['params']:
      index_by_legacy_name[legacy_names[position]] = index
      position += 1

  name_map = recover_name_map(
    legacy_model,
    legacy_names,
    attn_layers=attn_layers,
    n_layers=n_layers,
    num_heads=num_heads,
  )
  fla_names = set(_flatten(fla_groups))
  unmapped = fla_names - name_map.keys()
  if unmapped:
    raise ValueError(f'translate_model produces no tensor for FLA parameters {sorted(unmapped)}.')

  legacy_group_by_name = _group_index_by_name(legacy_groups)
  fla_group_by_name = _group_index_by_name(fla_groups)
  for fla_name in sorted(fla_names):
    source = name_map[fla_name]
    if source is None:
      # A synthesized bias is a no-decay parameter whose source weight decays.
      # That change of group is correct, so it is not compared.
      continue
    if fla_group_by_name[fla_name] != legacy_group_by_name[source]:
      raise ValueError(
        f'FLA parameter {fla_name!r} is in weight decay group {fla_group_by_name[fla_name]} '
        f'but its legacy source {source!r} is in group {legacy_group_by_name[source]}.'
      )

  source_state = {int(index): entry for index, entry in optimizer_state_dict['state'].items()}
  moment_keys = _check_adamw_state(source_state)

  translated_moments: dict[str, dict[str, Tensor]] = {}
  if source_state:
    missing_state = [name for name in legacy_names if index_by_legacy_name[name] not in source_state]
    if missing_state:
      raise ValueError(
        f'Optimizer state is missing entries for legacy parameters {sorted(missing_state)}. '
        'A partially populated AdamW state cannot be translated.'
      )
    for key in moment_keys:
      moments = {name: source_state[index][key] for name, index in index_by_legacy_name.items()}
      translated_moments[key] = translate_model(
        add_tied_lm_head(moments),
        attn_layers=attn_layers,
        n_layers=n_layers,
        num_heads=num_heads,
      )

  fla_index_by_name = {name: index for index, name in enumerate(_flatten(fla_groups))}
  state: dict[int, dict[str, Any]] = {}
  if source_state:
    for fla_name, fla_index in fla_index_by_name.items():
      source = name_map[fla_name]
      step_source = _synthesized_source(fla_name, name_map) if source is None else source
      entry: dict[str, Any] = {}
      source_entry = source_state[index_by_legacy_name[step_source]]
      if 'step' in source_entry:
        step = source_entry['step']
        entry['step'] = step.clone() if isinstance(step, torch.Tensor) else step
      for key in moment_keys:
        entry[key] = translated_moments[key][fla_name]
      state[fla_index] = entry

  param_groups = []
  offset = 0
  for source_group, names in zip(source_groups, fla_groups, strict=True):
    group = {key: value for key, value in source_group.items() if key != 'params'}
    group['params'] = list(range(offset, offset + len(names)))
    offset += len(names)
    param_groups.append(group)

  return {'state': state, 'param_groups': param_groups}

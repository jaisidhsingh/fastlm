from __future__ import annotations

import math
from types import SimpleNamespace

import torch
from torch import nn
from transformers import AutoModelForCausalLM

from src.models import builder
from src.models.legacy.construct import get_param_groups
from src.models.legacy.transformer import Transformer
from src.models.translate_backend import translate_model
from src.optim.translate_backend import (
  add_tied_lm_head,
  ordered_param_names,
  recover_name_map,
  translate_adamw,
)
from testing.translated_module_tests import (
  DEVICE,
  _legacy_model_config,
  _make_model_config,
  _normalize_config,
  _relative,
  _rms,
)

WEIGHT_DECAY = 0.1
LEARNING_RATE = 1e-3


def _attn_layers(config: SimpleNamespace) -> list[int]:
  _, ratio = builder.parse_arch_id(config.arch_id)
  if config.arch_id == 'attn':
    return list(range(config.n_layers))
  if ratio is None:
    return []
  return builder.build_hybrid_layers(config.n_layers, ratio)


def _build_models(config: SimpleNamespace, seed: int = 0) -> tuple[nn.Module, nn.Module]:
  torch.manual_seed(seed)
  legacy_model = Transformer(_legacy_model_config(config)).to(device=DEVICE, dtype=torch.float32)
  legacy_model.train()

  fla_model = AutoModelForCausalLM.from_config(builder.config_builder(config)).to(device=DEVICE, dtype=torch.float32)
  fla_model.load_state_dict(
    translate_model(
      legacy_model.state_dict(),
      attn_layers=_attn_layers(config),
      n_layers=config.n_layers,
      num_heads=config.n_heads,
    ),
    strict=True,
  )
  fla_model.train()
  return legacy_model, fla_model


def _build_adamw(model: nn.Module) -> torch.optim.AdamW:
  return torch.optim.AdamW(
    get_param_groups(model, WEIGHT_DECAY),
    lr=LEARNING_RATE,
    betas=[0.9, 0.95],
    weight_decay=WEIGHT_DECAY,
    eps=1e-8,
  )


def _batches(config: SimpleNamespace, count: int, seed: int = 1) -> list[torch.Tensor]:
  generator = torch.Generator(device='cpu').manual_seed(seed)
  return [
    torch.randint(low=0, high=config.vocab_size, size=(2, config.seq_len), generator=generator).to(DEVICE)
    for _ in range(count)
  ]


def _backward(model: nn.Module, optimizer: torch.optim.Optimizer, batch: torch.Tensor, *, fla_backend: bool) -> float:
  """Run the forward and backward the way `TorchEngine.step` runs them.

  Parameters stay in float32 and the forward runs under bfloat16 autocast. FLA
  attention calls flash-attn, which accepts only fp16 and bf16, so a float32
  forward is not an option; autocast is also what pretraining runs.
  """
  optimizer.zero_grad(set_to_none=True)
  with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    logits = model(batch).logits if fla_backend else model(batch)
  loss = nn.functional.cross_entropy(logits[:, :-1].flatten(0, 1).float(), batch[:, 1:].flatten())
  loss.backward()
  return loss.item()


def _step(model: nn.Module, optimizer: torch.optim.Optimizer, batch: torch.Tensor, *, fla_backend: bool) -> float:
  loss = _backward(model, optimizer, batch, fla_backend=fla_backend)
  optimizer.step()
  return loss


ALLCLOSE_TOLERANCES = ((1e-2, 1e-6), (1e-2, 1e-5), (5e-2, 1e-5), (1e-1, 1e-5))


def _summarize(
  label: str,
  actual: dict[str, torch.Tensor],
  expected: dict[str, torch.Tensor],
  name_map: dict[str, str | None],
) -> None:
  """Report how far a set of translated tensors is from the legacy set.

  Synthesized parameters are skipped: `expected` holds zeros for them, so they
  have no reference to be relative to. The aggregate pools every tensor, so it
  is not dominated by the small tensors the worst case always lands on.
  """
  names = [name for name in sorted(expected) if name in actual and name_map.get(name) is not None]
  difference_square = 0.0
  reference_square = 0.0
  equal_count = 0
  worst_name = None
  worst_relative = -1.0
  allclose_counts = [0] * len(ALLCLOSE_TOLERANCES)

  for name in names:
    difference = (actual[name].float() - expected[name].float()).flatten()
    reference = expected[name].float().flatten()
    difference_square += difference.pow(2).sum().item()
    reference_square += reference.pow(2).sum().item()
    if torch.equal(actual[name], expected[name]):
      equal_count += 1
    for index, (rtol, atol) in enumerate(ALLCLOSE_TOLERANCES):
      if torch.allclose(actual[name], expected[name], rtol=rtol, atol=atol):
        allclose_counts[index] += 1
    relative = _relative(_rms(difference), _rms(reference))
    if relative > worst_relative:
      worst_relative = relative
      worst_name = name

  aggregate = _relative(math.sqrt(difference_square), math.sqrt(reference_square))
  print(f'{label}: {len(names)} tensors')
  print(f'  aggregate relative rms: {aggregate:.6f}')
  print(f'  worst tensor: {worst_name}, relative rms {worst_relative:.6f}')
  print(f'  equal: {equal_count}/{len(names)}')
  for (rtol, atol), count in zip(ALLCLOSE_TOLERANCES, allclose_counts, strict=True):
    print(f'  allclose rtol={rtol:g} atol={atol:g}: {count}/{len(names)}')


def _clone_optimizer_state(state_dict: dict) -> dict:
  return {
    'state': {
      index: {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in entry.items()}
      for index, entry in state_dict['state'].items()
    },
    'param_groups': [dict(group) for group in state_dict['param_groups']],
  }


def _injected_updates(
  config: SimpleNamespace,
  *,
  optimizer_state: dict,
  legacy_before: dict[str, torch.Tensor],
  gradients: dict[str, torch.Tensor],
  attn_layers: list[int],
) -> dict[str, torch.Tensor]:
  """Step a translated model on the legacy gradients instead of its own.

  This removes the only remaining difference between the two backends at the
  resumed step. What is left is the optimizer state and the group
  hyperparameters, so any difference the caller measures is a translation
  error, not a kernel difference.

  `legacy_before` holds the legacy weights from before its own third step, so
  the injected model starts where the legacy model started.
  """
  _, model = _build_models(config)
  model.load_state_dict(
    translate_model(
      add_tied_lm_head(legacy_before),
      attn_layers=attn_layers,
      n_layers=config.n_layers,
      num_heads=config.n_heads,
    ),
    strict=True,
  )
  optimizer = _build_adamw(model)
  optimizer.load_state_dict(_clone_optimizer_state(optimizer_state))

  before = {}
  for name, parameter in model.named_parameters():
    parameter.grad = gradients[name].detach().clone().to(parameter.dtype)
    before[name] = parameter.detach().clone()
  optimizer.step()
  return {name: parameter.detach() - before[name] for name, parameter in model.named_parameters()}


def test_index_recovery(config: SimpleNamespace) -> None:
  legacy_model, fla_model = _build_models(config)
  for label, model in (('legacy', legacy_model), ('fla', fla_model)):
    groups = ordered_param_names(model, WEIGHT_DECAY)
    parameter_by_name = dict(model.named_parameters())
    for group_index, (group, names) in enumerate(zip(get_param_groups(model, WEIGHT_DECAY), groups, strict=True)):
      assert len(group['params']) == len(names), f'{label} group {group_index} length mismatch'
      for parameter, name in zip(group['params'], names, strict=True):
        assert parameter_by_name[name] is parameter, f'{label} group {group_index} name {name} is not the parameter'
    print(f'{label}: {[len(names) for names in groups]} parameters per group, all resolved by identity')


def test_probe_is_constant(config: SimpleNamespace) -> None:
  legacy_model, fla_model = _build_models(config)
  legacy_names = [name for group in ordered_param_names(legacy_model, WEIGHT_DECAY) for name in group]
  name_map = recover_name_map(
    legacy_model,
    legacy_names,
    attn_layers=_attn_layers(config),
    n_layers=config.n_layers,
    num_heads=config.n_heads,
  )
  fla_names = {name for group in ordered_param_names(fla_model, WEIGHT_DECAY) for name in group}
  synthesized = sorted(name for name in fla_names if name_map[name] is None)
  print(f'{len(name_map)} mapped names, synthesized: {synthesized}')


def test_two_step_equivalence(config: SimpleNamespace) -> None:
  """Translate weights and optimizer after two steps, then compare a third step.

  A control runs all three steps on a second FLA model that was translated
  before any step was taken, so it needs no optimizer translation. The control
  measures how far the two backends drift on their own. A translated resume that
  matches the control is as close to the legacy run as the backends allow.
  """
  legacy_model, fla_model = _build_models(config)
  _, control_model = _build_models(config)
  legacy_optimizer = _build_adamw(legacy_model)
  control_optimizer = _build_adamw(control_model)

  batches = _batches(config, 3)
  for batch in batches[:2]:
    _step(legacy_model, legacy_optimizer, batch, fla_backend=False)
    _step(control_model, control_optimizer, batch, fla_backend=True)

  attn_layers = _attn_layers(config)
  fla_model.load_state_dict(
    translate_model(
      legacy_model.state_dict(),
      attn_layers=attn_layers,
      n_layers=config.n_layers,
      num_heads=config.n_heads,
    ),
    strict=True,
  )
  translated_state = translate_adamw(
    legacy_optimizer.state_dict(),
    legacy_model=legacy_model,
    fla_model=fla_model,
    weight_decay=WEIGHT_DECAY,
    attn_layers=attn_layers,
    n_layers=config.n_layers,
    num_heads=config.n_heads,
  )
  # The translated tensors alias the legacy moments, which the legacy third
  # step overwrites in place. Clone before either optimizer loads them.
  translated_state = _clone_optimizer_state(translated_state)
  fla_optimizer = _build_adamw(fla_model)
  fla_optimizer.load_state_dict(_clone_optimizer_state(translated_state))

  legacy_names = [name for group in ordered_param_names(legacy_model, WEIGHT_DECAY) for name in group]
  name_map = recover_name_map(
    legacy_model,
    legacy_names,
    attn_layers=attn_layers,
    n_layers=config.n_layers,
    num_heads=config.n_heads,
  )

  fla_state = {
    name: fla_optimizer.state[parameter]
    for name, parameter in fla_model.named_parameters()
    if parameter in fla_optimizer.state
  }
  for key in ('exp_avg', 'exp_avg_sq'):
    loaded = {name: entry[key] for name, entry in fla_state.items()}
    legacy_moments = {
      name: legacy_optimizer.state[parameter][key] for name, parameter in legacy_model.named_parameters()
    }
    expected_moments = translate_model(
      add_tied_lm_head(legacy_moments),
      attn_layers=attn_layers,
      n_layers=config.n_layers,
      num_heads=config.n_heads,
    )
    for name, tensor in loaded.items():
      assert torch.equal(tensor, expected_moments[name]), f'AdamW holds a different {key} for {name} after loading'
  print(f'{len(fla_state)} loaded moments match the translated state exactly')

  legacy_loss = _backward(legacy_model, legacy_optimizer, batches[2], fla_backend=False)
  fla_loss = _backward(fla_model, fla_optimizer, batches[2], fla_backend=True)
  control_loss = _backward(control_model, control_optimizer, batches[2], fla_backend=True)
  print(f'third step loss: legacy {legacy_loss:.6f}, translated {fla_loss:.6f}, control {control_loss:.6f}')

  expected_gradients = translate_model(
    add_tied_lm_head({name: parameter.grad for name, parameter in legacy_model.named_parameters()}),
    attn_layers=attn_layers,
    n_layers=config.n_layers,
    num_heads=config.n_heads,
  )
  actual_gradients = {name: parameter.grad for name, parameter in fla_model.named_parameters()}
  _summarize('third step gradient', actual_gradients, expected_gradients, name_map)

  before = {name: parameter.detach().clone() for name, parameter in fla_model.named_parameters()}
  legacy_before = {name: parameter.detach().clone() for name, parameter in legacy_model.named_parameters()}
  control_before = {name: parameter.detach().clone() for name, parameter in control_model.named_parameters()}

  legacy_optimizer.step()
  fla_optimizer.step()
  control_optimizer.step()

  expected_updates = translate_model(
    add_tied_lm_head(
      {name: parameter.detach() - legacy_before[name] for name, parameter in legacy_model.named_parameters()}
    ),
    attn_layers=attn_layers,
    n_layers=config.n_layers,
    num_heads=config.n_heads,
  )
  actual_updates = {name: parameter.detach() - before[name] for name, parameter in fla_model.named_parameters()}
  _summarize('third step update', actual_updates, expected_updates, name_map)

  _summarize('third step update, legacy gradients injected', _injected_updates(
    config,
    optimizer_state=translated_state,
    legacy_before=legacy_before,
    gradients=expected_gradients,
    attn_layers=attn_layers,
  ), expected_updates, name_map)

  control_updates = {
    name: parameter.detach() - control_before[name] for name, parameter in control_model.named_parameters()
  }
  _summarize('third step update, control', control_updates, expected_updates, name_map)

  expected = translate_model(
    legacy_model.state_dict(),
    attn_layers=attn_layers,
    n_layers=config.n_layers,
    num_heads=config.n_heads,
  )
  _summarize('parameter after the third step', fla_model.state_dict(), expected, name_map)
  _summarize('parameter after the third step, control', control_model.state_dict(), expected, name_map)

  for name in sorted(expected):
    if name_map.get(name) is None and name in fla_model.state_dict():
      print(f'{name}: no legacy counterpart, rms after one FLA step {_rms(fla_model.state_dict()[name])}')


def test_moment_placement(config: SimpleNamespace) -> None:
  """Check that each translated moment sits on the FLA parameter its source feeds.

  The FFN split applies to every architecture, so the two backends do not hold
  the same number of parameters and the indices cannot be compared one to one.
  This compares each FLA index against `translate_model` run on the moments
  directly, which is the check that the index map is not off by a position.
  """
  legacy_model, fla_model = _build_models(config)
  legacy_optimizer = _build_adamw(legacy_model)
  for batch in _batches(config, 2):
    _step(legacy_model, legacy_optimizer, batch, fla_backend=False)

  attn_layers = _attn_layers(config)
  source = legacy_optimizer.state_dict()
  translated = translate_adamw(
    source,
    legacy_model=legacy_model,
    fla_model=fla_model,
    weight_decay=WEIGHT_DECAY,
    attn_layers=attn_layers,
    n_layers=config.n_layers,
    num_heads=config.n_heads,
  )

  legacy_names = [name for group in ordered_param_names(legacy_model, WEIGHT_DECAY) for name in group]
  source_index_by_name = {}
  position = 0
  for group in source['param_groups']:
    for index in group['params']:
      source_index_by_name[legacy_names[position]] = index
      position += 1

  fla_names = [name for group in ordered_param_names(fla_model, WEIGHT_DECAY) for name in group]
  fla_index_by_name = {name: index for index, name in enumerate(fla_names)}

  for key in ('exp_avg', 'exp_avg_sq'):
    moments = {name: source['state'][index][key] for name, index in source_index_by_name.items()}
    if 'lm_head.weight' not in moments:
      moments['lm_head.weight'] = moments['embed_tokens.weight']
    expected = translate_model(
      moments,
      attn_layers=attn_layers,
      n_layers=config.n_layers,
      num_heads=config.n_heads,
    )
    for name, index in fla_index_by_name.items():
      assert torch.equal(translated['state'][index][key], expected[name]), f'{name} {key} is on the wrong index'

  identity_pairs = [
    ('model.embeddings.weight', 'embed_tokens.weight'),
    ('model.norm.weight', 'out_norm.weight'),
  ]
  for layer in range(config.n_layers):
    identity_pairs.append((f'model.layers.{layer}.attn_norm.weight', f'layers.{layer}.token_mixer_norm.weight'))
    identity_pairs.append((f'model.layers.{layer}.mlp_norm.weight', f'layers.{layer}.mlp_norm.weight'))
    identity_pairs.append((f'model.layers.{layer}.mlp.down_proj.weight', f'layers.{layer}.mlp.fc2.weight'))
    if layer in attn_layers:
      identity_pairs.append((f'model.layers.{layer}.attn.o_proj.weight', f'layers.{layer}.token_mixer.w_out.weight'))
      continue
    for suffix in ('q_proj.weight', 'k_proj.weight', 'v_proj.weight', 'a_proj.weight', 'b_proj.weight', 'A_log',
                   'dt_bias', 'o_norm.weight', 'o_proj.weight'):
      identity_pairs.append((f'model.layers.{layer}.attn.{suffix}', f'layers.{layer}.token_mixer.{suffix}'))

  for fla_name, legacy_name in identity_pairs:
    if fla_name not in fla_index_by_name:
      raise AssertionError(f'{fla_name} is not an FLA parameter')
    target = translated['state'][fla_index_by_name[fla_name]]
    origin = source['state'][source_index_by_name[legacy_name]]
    for key in ('exp_avg', 'exp_avg_sq'):
      assert torch.equal(target[key], origin[key]), f'{fla_name} {key} does not equal {legacy_name} {key}'
    assert int(target['step']) == int(origin['step']), f'{fla_name} step does not equal {legacy_name} step'

  print(f'{len(fla_names)} indices placed correctly, {len(identity_pairs)} untransformed moments equal exactly')


def test_gate_bias(config: SimpleNamespace) -> None:
  legacy_model, fla_model = _build_models(config)
  legacy_optimizer = _build_adamw(legacy_model)
  for batch in _batches(config, 2):
    _step(legacy_model, legacy_optimizer, batch, fla_backend=False)

  attn_layers = _attn_layers(config)
  translated = translate_adamw(
    legacy_optimizer.state_dict(),
    legacy_model=legacy_model,
    fla_model=fla_model,
    weight_decay=WEIGHT_DECAY,
    attn_layers=attn_layers,
    n_layers=config.n_layers,
    num_heads=config.n_heads,
  )

  fla_groups = ordered_param_names(fla_model, WEIGHT_DECAY)
  fla_names = [name for group in fla_groups for name in group]
  index_by_name = {name: index for index, name in enumerate(fla_names)}
  group_by_name = {name: group_index for group_index, group in enumerate(fla_groups) for name in group}

  legacy_groups = ordered_param_names(legacy_model, WEIGHT_DECAY)
  legacy_group_by_name = {name: group_index for group_index, group in enumerate(legacy_groups) for name in group}

  checked = 0
  for layer in attn_layers:
    bias_name = f'model.layers.{layer}.attn.gate.bias'
    if bias_name not in index_by_name:
      continue
    entry = translated['state'][index_by_name[bias_name]]
    assert torch.count_nonzero(entry['exp_avg']) == 0, f'{bias_name} exp_avg is not zero'
    assert torch.count_nonzero(entry['exp_avg_sq']) == 0, f'{bias_name} exp_avg_sq is not zero'
    reference = translated['state'][index_by_name[f'model.layers.{layer}.attn.q_proj.weight']]
    assert int(entry['step']) == int(reference['step']), f'{bias_name} step is not the layer step'
    assert group_by_name[bias_name] == 1, f'{bias_name} is not in the no decay group'
    assert legacy_group_by_name[f'layers.{layer}.token_mixer.w_gate.weight'] == 0, 'w_gate.weight does not decay'
    checked += 1
  print(f'{checked} gate biases checked: zero moments, step {int(entry["step"])}, no decay group')


def test_rejection(config: SimpleNamespace) -> None:
  legacy_model, fla_model = _build_models(config)
  attn_layers = _attn_layers(config)
  batch = _batches(config, 1)[0]

  def translate(state_dict):
    return translate_adamw(
      state_dict,
      legacy_model=legacy_model,
      fla_model=fla_model,
      weight_decay=WEIGHT_DECAY,
      attn_layers=attn_layers,
      n_layers=config.n_layers,
      num_heads=config.n_heads,
    )

  sgd = torch.optim.SGD(get_param_groups(legacy_model, WEIGHT_DECAY), lr=LEARNING_RATE, momentum=0.9)
  _step(legacy_model, sgd, batch, fla_backend=False)
  try:
    translate(sgd.state_dict())
  except ValueError as error:
    print(f'sgd rejected: {error}')
  else:
    raise AssertionError('sgd state was not rejected')

  adamw = _build_adamw(legacy_model)
  _step(legacy_model, adamw, batch, fla_backend=False)
  truncated = adamw.state_dict()
  truncated['param_groups'] = [
    {**truncated['param_groups'][0], 'params': truncated['param_groups'][0]['params'][:-1]},
    truncated['param_groups'][1],
  ]
  try:
    translate(truncated)
  except ValueError as error:
    print(f'truncated param_groups rejected: {error}')
  else:
    raise AssertionError('a truncated param_groups was not rejected')

  try:
    import schedulefree
  except ImportError:
    print('schedulefree is not installed, skipping the sfo_adamw rejection check')
    return

  sfo = schedulefree.AdamWScheduleFree(get_param_groups(legacy_model, WEIGHT_DECAY), lr=LEARNING_RATE, warmup_steps=1)
  sfo.train()
  _step(legacy_model, sfo, batch, fla_backend=False)
  try:
    translate(sfo.state_dict())
  except ValueError as error:
    print(f'sfo_adamw rejected: {error}')
  else:
    raise AssertionError('sfo_adamw state was not rejected')


def main() -> None:
  configs = {
    'attn': _make_model_config('attn', 1),
    'gdn': _make_model_config('gdn', 1),
    'gdn+attn_1-1': _make_model_config('gdn+attn', 1),
  }

  for arch_id, config in configs.items():
    print(f'=== {arch_id}: index recovery ===')
    test_index_recovery(config)
    print(f'=== {arch_id}: probe is constant ===')
    test_probe_is_constant(config)
    print(f'=== {arch_id}: two step equivalence ===')
    test_two_step_equivalence(config)
    print(f'=== {arch_id}: moment placement ===')
    test_moment_placement(config)
    if arch_id != 'gdn':
      print(f'=== {arch_id}: gate bias ===')
      test_gate_bias(config)
    print()

  print('=== attn: rejection ===')
  test_rejection(_normalize_config(configs['attn']))


if __name__ == '__main__':
  main()

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import NamedTuple

import torch
import tyro
from torch import nn
from transformers import AutoModelForCausalLM

from src.models import builder
from src.models.legacy.construct import get_param_groups
from src.models.legacy.transformer import Transformer
from src.models.translate_backend import translate_model
from src.optim.translate_backend import translate_adamw
from testing.translated_module_tests import DEVICE, _legacy_model_config, _normalize_config


class Hyperparameters(NamedTuple):
  lr: float
  weight_decay: float
  beta1: float
  beta2: float
  eps: float
  fused: bool


@dataclass
class Config:
  ckpt_path: str | None = None
  steps: int = 3
  resume_step: int | None = None
  batch_size: int = 2
  batch_seed: int = 1
  no_baseline: bool = False
  arch_id: str = 'gdn+attn_1-1'
  n_layers: int = 4
  dim: int = 64
  n_heads: int = 4
  seq_len: int = 128
  vocab_size: int = 256
  expand: str = '4'
  seed: int = 0
  lr: float = 1e-3
  weight_decay: float = 0.1
  beta1: float = 0.9
  beta2: float = 0.95
  eps: float = 1e-8
  fused: bool = False


def parse_args() -> Config:
  config = tyro.cli(Config)
  if config.steps < 1:
    raise ValueError(f'steps must be at least 1, got {config.steps}')
  if config.resume_step is None:
    config.resume_step = 0 if config.ckpt_path is not None else config.steps - 1
  if not 0 <= config.resume_step < config.steps:
    raise ValueError(f'resume_step must be in range(0, {config.steps}), got {config.resume_step}')
  return config


def build_config(args: Config) -> SimpleNamespace:
  arch, ratio = builder.parse_arch_id(args.arch_id)
  return _normalize_config(
    dict(
      arch_id=args.arch_id,
      token_mixer=arch,
      hybrid_mixer_ratio=1 if ratio is None else ratio,
      vocab_size=args.vocab_size,
      dim=args.dim,
      n_heads=args.n_heads,
      n_layers=args.n_layers,
      seq_len=args.seq_len,
      expand=args.expand,
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


def config_from_checkpoint(values: dict) -> SimpleNamespace:
  """Rebuild the model config the checkpoint was written with.

  `save_checkpoint` stores the whole run config under 'config', so nothing here
  has to be guessed or passed in again. Only two fields are overridden:
  `model_dtype`, because this script keeps parameters in float32 and runs the
  forward under autocast, and `intra_doc`, because random batches carry no
  document boundaries.
  """
  arch, ratio = builder.parse_arch_id(values['arch_id'])
  return _normalize_config(
    dict(
      arch_id=values['arch_id'],
      token_mixer=values.get('token_mixer', arch),
      hybrid_mixer_ratio=values.get('hybrid_mixer_ratio', 1 if ratio is None else ratio),
      vocab_size=values['vocab_size'],
      dim=values['d_model'],
      n_heads=values['n_heads'],
      n_layers=values['n_layers'],
      seq_len=values['seq_len'],
      expand=values['expand'],
      mlp=values['mlp_class'],
      rmsnorm_eps=values.get('rmsnorm_eps', 1e-6),
      model_dtype='float32',
      tie_embeddings=values['tie_embeddings'],
      attn_gate=values['attn_gate'],
      attn_qk_norm=values['attn_qk_norm'],
      gdn_conv_size=values['gdn_conv_size'],
      gdn_gate=values['gdn_gate'],
      gdn_neg_eigval=values['gdn_neg_eigval'],
      intra_doc=False,
    )
  )


def hyperparameters_from_args(args: Config) -> Hyperparameters:
  return Hyperparameters(
    lr=args.lr,
    weight_decay=args.weight_decay,
    beta1=args.beta1,
    beta2=args.beta2,
    eps=args.eps,
    fused=args.fused,
  )


def hyperparameters_from_checkpoint(values: dict, optimizer_state: dict) -> Hyperparameters:
  """Read the AdamW settings in force at the checkpoint.

  These come from the saved `param_groups`, not from the run config. The config
  holds the peak learning rate, while the group holds the one the scheduler had
  reached, and for a decayed checkpoint the two differ by two orders of
  magnitude. `Optimizer.load_state_dict` overwrites the group with the saved
  value anyway, so the config learning rate would silently apply to the
  weights_only baseline alone and make it incomparable.

  `fused` has to match too. `load_state_dict` copies every group hyperparameter
  out of the state dict, so an optimizer built one way and loaded from the other
  would step through a path it was not constructed for.
  """
  if values['optim'] != 'adamw':
    raise ValueError(f'Only AdamW can be translated, but the checkpoint used optim={values["optim"]!r}.')

  group = optimizer_state['param_groups'][0]
  weight_decay = values['weight_decay']
  if group['weight_decay'] != weight_decay:
    raise ValueError(
      f'The decay group holds weight_decay={group["weight_decay"]}, but the config says {weight_decay}. '
      'get_param_groups needs the value the run used to reproduce the same grouping.'
    )
  beta1, beta2 = group['betas']
  return Hyperparameters(
    lr=group['lr'],
    weight_decay=weight_decay,
    beta1=beta1,
    beta2=beta2,
    eps=group['eps'],
    fused=group.get('fused', False) or False,
  )


def attn_layers_of(config: SimpleNamespace) -> list[int]:
  """List the layer indices holding an attention token mixer.

  FLA names the token mixer `attn` for both layer types, so this cannot be
  recovered from the key names and `translate_model` takes it as an argument.
  """
  _, ratio = builder.parse_arch_id(config.arch_id)
  if config.arch_id == 'attn':
    return list(range(config.n_layers))
  if ratio is None:
    return []
  return builder.build_hybrid_layers(config.n_layers, ratio)


def build_legacy_model(config: SimpleNamespace, seed: int) -> nn.Module:
  torch.manual_seed(seed)
  model = Transformer(_legacy_model_config(config)).to(device=DEVICE, dtype=torch.float32)
  model.train()
  return model


def build_fla_model(config: SimpleNamespace, legacy_model: nn.Module) -> nn.Module:
  model = AutoModelForCausalLM.from_config(builder.config_builder(config)).to(device=DEVICE, dtype=torch.float32)
  model.load_state_dict(
    translate_model(
      legacy_model.state_dict(),
      attn_layers=attn_layers_of(config),
      n_layers=config.n_layers,
      num_heads=config.n_heads,
    ),
    strict=True,
  )
  model.train()
  return model


def build_adamw(model: nn.Module, hparams: Hyperparameters) -> torch.optim.AdamW:
  return torch.optim.AdamW(
    get_param_groups(model, hparams.weight_decay),
    lr=hparams.lr,
    betas=[hparams.beta1, hparams.beta2],
    weight_decay=hparams.weight_decay,
    eps=hparams.eps,
    fused=hparams.fused,
  )


def clone_optimizer_state(state_dict: dict) -> dict:
  """Copy every state tensor of an optimizer state dict.

  `translate_adamw` returns moments that share storage with the source
  checkpoint, and `AdamW.load_state_dict` keeps them whenever dtype and device
  already match. Without this copy the legacy optimizer and the translated one
  write the same tensors, and each step of one corrupts the other.
  """
  return {
    'state': {
      index: {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in entry.items()}
      for index, entry in state_dict['state'].items()
    },
    'param_groups': [dict(group) for group in state_dict['param_groups']],
  }


def load_checkpoint(path: str) -> dict:
  checkpoint = torch.load(path, map_location='cpu', weights_only=False)
  for key in ('config', 'state_dict', 'optimizer'):
    if key not in checkpoint:
      raise ValueError(f'Checkpoint {path} has no {key!r}.')
  if checkpoint['optimizer'] is None:
    raise ValueError(f'Checkpoint {path} was written with save_optim=False, so it holds no optimizer state.')
  # `apply_compile` compiles each block, so `_orig_mod.` lands inside the key,
  # as in `layers.0._orig_mod.token_mixer.A_log`, not only at the front. This is
  # the same replace `maybe_load_checkpoint` runs. `get_param_groups` runs on the
  # uncompiled model, so the optimizer indices carry no prefix and need nothing.
  checkpoint['state_dict'] = {key.replace('_orig_mod.', ''): value for key, value in checkpoint['state_dict'].items()}
  return checkpoint


def build_legacy_from_checkpoint(
  checkpoint: dict,
  config: SimpleNamespace,
  hparams: Hyperparameters,
) -> tuple[nn.Module, torch.optim.AdamW]:
  model = Transformer(_legacy_model_config(config)).to(device=DEVICE, dtype=torch.float32)
  model.load_state_dict(checkpoint['state_dict'], strict=True)
  model.train()
  optimizer = build_adamw(model, hparams)
  optimizer.load_state_dict(clone_optimizer_state(checkpoint['optimizer']))
  return model, optimizer


def build_translated(
  legacy_model: nn.Module,
  legacy_optimizer: torch.optim.Optimizer,
  config: SimpleNamespace,
  hparams: Hyperparameters,
) -> tuple[nn.Module, torch.optim.AdamW]:
  model = build_fla_model(config, legacy_model)
  optimizer = build_adamw(model, hparams)
  optimizer.load_state_dict(
    clone_optimizer_state(
      translate_adamw(
        legacy_optimizer.state_dict(),
        legacy_model=legacy_model,
        fla_model=model,
        weight_decay=hparams.weight_decay,
        attn_layers=attn_layers_of(config),
        n_layers=config.n_layers,
        num_heads=config.n_heads,
      )
    )
  )
  return model, optimizer


def build_weights_only(
  legacy_model: nn.Module,
  config: SimpleNamespace,
  hparams: Hyperparameters,
) -> tuple[nn.Module, torch.optim.AdamW]:
  model = build_fla_model(config, legacy_model)
  return model, build_adamw(model, hparams)


def make_batches(config: SimpleNamespace, args: Config) -> list[torch.Tensor]:
  generator = torch.Generator(device='cpu').manual_seed(args.batch_seed)
  if config.seq_len != 2048:
      config.seq_len = 2048
  else:
      print("Sequence length is already 2048")

  return [
    torch.randint(
      low=0,
      high=config.vocab_size,
      size=(args.batch_size, config.seq_len),
      generator=generator,
    ).to(DEVICE)
    for _ in range(args.steps)
  ]


def step(
  model: nn.Module,
  optimizer: torch.optim.Optimizer,
  batch: torch.Tensor,
  *,
  fla_backend: bool,
) -> float:
  optimizer.zero_grad(set_to_none=True)
  with torch.amp.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
    logits = model(batch).logits if fla_backend else model(batch)
  loss = nn.functional.cross_entropy(logits[:, :-1].flatten(0, 1).float(), batch[:, 1:].flatten())
  loss.backward()
  optimizer.step()
  return loss.item()


def _relative(difference: float, reference: float) -> float:
  if reference != 0.0:
    return abs(difference) / abs(reference)
  return 0.0 if difference == 0.0 else float('inf')


def _format(value: float | None, width: int, precision: int) -> str:
  return ' ' * (width - 1) + '-' if value is None else f'{value:{width}.{precision}f}'


def _report(rows: list[tuple[int, float, float | None, float | None]], baseline_label: str) -> None:
  print()
  header = f'{"step":>4}  {"legacy":>12}  {"translated":>12}  {baseline_label:>12}'
  print(f'{header}  {"rel t-l":>12}  {"rel b-l":>12}')
  for number, legacy_loss, translated_loss, baseline_loss in rows:
    translated_relative = None if translated_loss is None else _relative(translated_loss - legacy_loss, legacy_loss)
    baseline_relative = None if baseline_loss is None else _relative(baseline_loss - legacy_loss, legacy_loss)
    print(
      f'{number:>4}  {legacy_loss:12.6f}  {_format(translated_loss, 12, 6)}  {_format(baseline_loss, 12, 6)}  '
      f'{_format(translated_relative, 12, 9)}  {_format(baseline_relative, 12, 9)}'
    )

  translated_relatives = [
    _relative(translated_loss - legacy_loss, legacy_loss)
    for _, legacy_loss, translated_loss, _ in rows
    if translated_loss is not None
  ]
  baseline_relatives = [
    _relative(baseline_loss - legacy_loss, legacy_loss)
    for _, legacy_loss, _, baseline_loss in rows
    if baseline_loss is not None
  ]

  print()
  if translated_relatives:
    print(f'worst relative loss difference, translated against legacy: {max(translated_relatives):.9f}')
  if baseline_relatives:
    print(f'worst relative loss difference, {baseline_label} against legacy: {max(baseline_relatives):.9f}')


def main() -> None:
  args = parse_args()

  if args.ckpt_path is not None:
    checkpoint = load_checkpoint(args.ckpt_path)
    values = checkpoint['config']
    config = config_from_checkpoint(values)
    hparams = hyperparameters_from_checkpoint(values, checkpoint['optimizer'])
    baseline_label = 'weights_only'
    print(f'checkpoint: {args.ckpt_path}, written at step {checkpoint.get("step")}')
    if values.get('intra_doc_masking', False):
      print('note: the run used intra_doc_masking, which is forced off here; the losses are not the run losses')
  else:
    checkpoint = None
    config = build_config(args)
    hparams = hyperparameters_from_args(args)
    baseline_label = 'control'

  print(
    f'arch_id: {config.arch_id}, layers: {config.n_layers}, dim: {config.dim}, heads: {config.n_heads}, '
    f'seq_len: {config.seq_len}, vocab: {config.vocab_size}'
  )
  print(f'attention layers: {attn_layers_of(config)}, device: {DEVICE}')
  print(
    f'adamw: lr={hparams.lr}, betas=({hparams.beta1}, {hparams.beta2}), '
    f'weight_decay={hparams.weight_decay}, eps={hparams.eps}, fused={hparams.fused}'
  )
  print(f'steps: {args.steps}, translation after legacy step {args.resume_step}, batch size {args.batch_size}')

  if checkpoint is not None:
    legacy_model, legacy_optimizer = build_legacy_from_checkpoint(checkpoint, config, hparams)
  else:
    legacy_model = build_legacy_model(config, args.seed)
    legacy_optimizer = build_adamw(legacy_model, hparams)

  baseline_model = None
  baseline_optimizer = None
  if not args.no_baseline:
    # Both labels are the same construction, an FLA model on the current legacy
    # weights with a fresh optimizer. Without a checkpoint it is built before any
    # step, so it goes on to run the whole history itself.
    baseline_model, baseline_optimizer = build_weights_only(legacy_model, config, hparams)

  translated_model = None
  translated_optimizer = None
  if args.resume_step == 0:
    translated_model, translated_optimizer = build_translated(legacy_model, legacy_optimizer, config, hparams)

  batches = make_batches(config, args)
  rows: list[tuple[int, float, float | None, float | None]] = []

  for index, batch in enumerate(batches):
    number = index + 1
    legacy_loss = step(legacy_model, legacy_optimizer, batch, fla_backend=False)

    translated_loss = None
    if translated_model is not None:
      translated_loss = step(translated_model, translated_optimizer, batch, fla_backend=True)

    baseline_loss = None
    if baseline_model is not None:
      baseline_loss = step(baseline_model, baseline_optimizer, batch, fla_backend=True)

    rows.append((number, legacy_loss, translated_loss, baseline_loss))

    if translated_model is None and number == args.resume_step:
      translated_model, translated_optimizer = build_translated(legacy_model, legacy_optimizer, config, hparams)

  _report(rows, baseline_label)


if __name__ == '__main__':
  main()

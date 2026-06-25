import json
import os
import re

import torch
from pandas._libs.lib import i8max

from src import utils
from src.constants import SCALING_LADDER


def create_save_steps(cfg, world_size):
  token_budget_ids_ref = list(SCALING_LADDER['batch_size_vs_token_budget_strategy']['staggered_grid'].keys())
  token_budget_ids_ref = [float(x[:-1]) for x in token_budget_ids_ref]
  token_budget_ids_ref.sort()
  token_budget_ids_ref = [str(x) + 'B' for x in token_budget_ids_ref]

  if not cfg.resume or (
    cfg.resume and not cfg.cooldown_only
  ):  # training from init/resume -> (warmup) + stable -> cooldown
    # we want to save the start of the cooldown for all the intermediate token_budget_ids,
    # and for the last one, i.e. cfg.token_budget_id, we want to save when the cooldown starts + the final ckpt.
    index = token_budget_ids_ref.index(cfg.token_budget_id)
    save_ids = token_budget_ids_ref[: index + 1]

    if cfg.resume:
      start_idx = token_budget_ids_ref.index(cfg.resume_exp_name.split('_')[-1].replace('p', '.'))
      save_ids = token_budget_ids_ref[start_idx + 1 :]

    points_to_save_at = [float(x[:-1]) * 1e9 for x in save_ids]
    per_step_tokens = cfg.micro_batch_size * cfg.grad_accumulation_steps * cfg.seq_len * world_size
    points_to_save_at = [t // per_step_tokens for t in points_to_save_at]

    cooldown_steps_per_point = [
      cfg.cooldown_steps if isinstance(cfg.cooldown_steps, int) else int(cfg.cooldown_steps * p)
      for p in points_to_save_at
    ]
    points_to_save_at = [t - c for t, c in zip(points_to_save_at, cooldown_steps_per_point)]
    names = []

    for i in range(len(points_to_save_at)):
      tok_id = save_ids[i]
      names.append(f'decay_starts_to_{tok_id.replace(".", "p")}')

    # we want to save the last one because we decay automatically at the end
    if not cfg.resume:
      points_to_save_at = [cfg.warmup_steps] + points_to_save_at
      names = ['warmup_done'] + names

    # when we resume, cfg.steps_budget becomes the gap between cfg.resume_step and the step at which we satisfy cfg.token_budget_id
    points_to_save_at = points_to_save_at + [cfg.steps_budget]
    names = names + [f'decayed_to_{cfg.token_budget_id.replace(".", "p")}']

    tokens_at_points = [t * per_step_tokens for t in points_to_save_at]
    return {int(k): v for k, v in zip(points_to_save_at, names)}, [round(tok / 1e9, 1) for tok in tokens_at_points]

  if cfg.resume and cfg.cooldown_only:  # training from resume -> cooldown
    cooldown_steps = (
      cfg.cooldown_steps if isinstance(cfg.cooldown_steps, int) else int(cfg.cooldown_steps * cfg.steps_budget)
    )
    tok = float(cfg.token_budget_id[:-1]) * 1e9
    return {cooldown_steps + cfg.resume_step: f'decayed_to_{cfg.token_budget_id.replace(".", "p")}'}, [
      round(tok / 1e9, 1)
    ]

  return None, None


def save_checkpoint(step, model, engine, cfg, metrics, name, world_size):
  optimizer = engine.optimizer
  scheduler = engine.scheduler
  scaler = engine.scaler

  save_optim = getattr(cfg, 'save_optim', True)
  save_scheduler = getattr(cfg, 'save_scheduler', True)
  save_scaler = getattr(cfg, 'save_scaler', True)

  try:
    config_dict = vars(cfg)
  except:
    config_dict = cfg._asdict()

  state = {
    'step': step,
    'config': config_dict,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict() if save_optim else None,
    'scheduler': scheduler.state_dict() if scheduler and save_scheduler else {},
    'scaler': scaler.state_dict() if save_scaler else None,
  }

  save_folder = utils.get_exp_dir_path(cfg, world_size)
  os.makedirs(save_folder, exist_ok=True)

  # Add info about the step at which we are saving
  with open(os.path.join(save_folder, 'info.txt'), 'a') as f:
    f.writelines([f'{name} = step_{step}\n'])

  save_path = os.path.join(save_folder, f'ckpt_{name}.pt')
  utils.print_master(f'Saving checkpoint to {save_path}')
  torch.save(state, save_path)

  metrics_path = os.path.join(save_folder, f'metrics_{name}.json')
  with open(metrics_path, 'w') as f:
    json.dump(dict(metrics), f)


def maybe_load_checkpoint(cfg, world_size):
  ckpt = None

  if cfg.resume:
    save_folder = utils.get_exp_dir_path(cfg, world_size)
    if cfg.cooldown_only:
      print('Only cooling down the learning rate.')
      ckpt_path = os.path.join(save_folder, f'ckpt_decay_starts_to_{cfg.token_budget_id.replace(".", "p")}.pt')
    else:
      print('Resuming training in stable learning rate region.')
      ckpt_path = os.path.join(save_folder, f'ckpt_{cfg.resume_exp_name}.pt')
    print(f'Loading checkpoint from {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

  return ckpt


def load_metrics_from_checkpoint(cfg, world_size):
  ckpt = None

  if cfg.resume:
    save_folder = utils.get_exp_dir_path(cfg, world_size)
    ckpt_path = os.path.join(save_folder, f'metrics_{cfg.resume_exp_name}.json')
    with open(ckpt_path) as f:
      ckpt = json.load(f)

  return ckpt


def match_state_dict_keys(state_dict: dict, state_dict_orig: dict) -> dict:
  """Modifies the keys of 'state_dict' to match the keys of 'state_dict_orig'.

  Takes care of stat_dict discrepancies caused by DDP or torch.compile,
  drop any prefixes from the state_dict, then add the correct prefixes in the correct order.

  Args:
      state_dict (dict): dict to modify
      state_dict_orig (dict): dict to match
  """

  state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
  state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

  orig_key = next(iter(state_dict_orig.keys()))  # first key of orig state_dict

  if orig_key.startswith('_orig_mod.module.'):
    state_dict = {'_orig_mod.module.' + k: v for k, v in state_dict.items()}
  elif orig_key.startswith('_orig_mod.'):
    state_dict = {'_orig_mod.' + k: v for k, v in state_dict.items()}
  elif orig_key.startswith('module._orig_mod.'):
    state_dict = {'module._orig_mod.' + k: v for k, v in state_dict.items()}
  elif orig_key.startswith('module.'):
    state_dict = {'module.' + k: v for k, v in state_dict.items()}

  return state_dict

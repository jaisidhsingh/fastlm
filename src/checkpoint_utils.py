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

  if not cfg.resume:
    # we want to save the start of the cooldown for all the intermediate token_budget_ids,
    # and for the last one, i.e. cfg.token_budget_id, we want to save when the cooldown starts + the final ckpt.
    index = token_budget_ids_ref.index(cfg.token_budget_id)
    save_ids = token_budget_ids_ref[: index + 2]

    points_to_save_at = [float(x[:-1]) * 1e9 for x in save_ids]
    points_to_save_at = [
      t // (cfg.micro_batch_size * cfg.grad_accumulation_steps * cfg.seq_len * world_size) for t in points_to_save_at
    ]
    points_to_save_at = [t - cfg.cooldown_steps for t in points_to_save_at]
    names = []

    for i in range(len(points_to_save_at)):
      tok_id = save_ids[i]
      names.append(f'decay_starts_to_{tok_id.replace(".", "p")}')

    # we want to save the last one because we decay automatically at the end
    points_to_save_at = [cfg.warmup_steps] + points_to_save_at + [cfg.steps_budget]
    names = ['warmup_done'] + [names] + [f'decayed_to_{cfg.token_budget_id.replace(".", ",")}']

    return {k: v for k, v in zip(points_to_save_at, names)}

  else:
    return {cfg.cooldown_steps: f'decayed_to_{cfg.token_budget_id.replace(".", ",")}'}


def save_checkpoint(step, model, engine, cfg, metrics, name):
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

  save_folder = utils.get_exp_dir_path(cfg)
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


def maybe_load_checkpoint(cfg):
  """Each job_idx will restore where it left of."""
  ckpt = None

  if cfg.resume:
    # Paths are saved with utils.get_exp_dir_path(cfg), with global flags we can call that again from here
    # but this will not control the job_idx name.
    # If cfg.resume_exp_name is given, we will resume from that experiment name
    # else, this will resume the exact sweep with only one config line!
    # commented out because maybe it is not straightforward logic with current config design
    # if cfg.resume_exp_name:
    #   ckpt_dir = os.path.join(cfg.out_dir, cfg.resume_exp_name)
    # else: # verbatim as it was saved
    #   ckpt_dir = utils.get_exp_dir_path(cfg)

    # notice that we can resume from `resume_exp_name`, but save to a different `exp_name`
    resume_exp_name = cfg.resume_exp_name if cfg.resume_exp_name is not None else cfg.exp_name
    ckpt_dir = os.path.join(cfg.out_dir, resume_exp_name)

    # resume from a specified checkpoint or from the latest
    if cfg.resume_step is not None:
      ckpt_path = os.path.join(ckpt_dir, f'ckpt_step_{cfg.resume_step}.pth')
    else:
      ckpt_path = _latest_checkpoint(ckpt_dir, prefix='ckpt_step_')

    # load checkpoint on cpu to later avoid OOM when intializing the model on device
    # (an alternative design would be to initialize the model on `meta` device instead)
    print(f'Loading checkpoint from {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location='cpu')

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

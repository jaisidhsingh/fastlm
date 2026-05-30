import math
import os
import shutil
from collections import namedtuple
from itertools import product
from types import SimpleNamespace

import torch
import wandb
import yaml
from absl import flags

from src.constants import DEFAULT_CONFIG, SCALING_LADDER
from src.engine import TorchEngine

FLAGS = flags.FLAGS


GPU_PEAK_FLOPS_PER_SEC_MAP = {
  'NVIDIA A100-SXM4-80GB': 312e12,
  'NVIDIA H100 80GB HBM3': 989e12,
  'NVIDIA H100': 835e12,
}


def get_steps_from_chinchilla_multiplier(cfg, non_embed_params: int, world_size: int) -> int:
  token_budget = cfg.chinchilla_token_multiplier * non_embed_params * 20
  return int(round(token_budget / (cfg.micro_batch_size * cfg.seq_len * cfg.grad_accumulation_steps * world_size)))


def get_steps_budget(cfg, engine: TorchEngine, non_embed_params: int, world_size: int) -> int:
  steps_budget = None
  if cfg.scheduler == 'linear_cooldown':
    steps_budget = cfg.resume_step + engine.scheduler.cooldown_steps
  elif cfg.steps_budget == -1:
    steps_budget = get_steps_from_chinchilla_multiplier(cfg, non_embed_params, world_size)
  else:
    steps_budget = cfg.steps_budget
  assert steps_budget
  return steps_budget


def load_config_from_constants(param_scale_id):
  config_dict = DEFAULT_CONFIG
  for k in ['d_model', 'n_layers', 'n_heads']:
    config_dict[k] = SCALING_LADDER['models'][param_scale_id][k]
  return SimpleNamespace(**config_dict)


def load_config(path):
  """
  Parse a yaml file and return the correspondent config as a namedtuple.
  If the config files has multiple entries, returns the one corresponding to job_idx.
  """

  with open(path, 'r') as file:
    config_dict = yaml.safe_load(file)
  Config = namedtuple('Config', config_dict.keys())

  if FLAGS.job_idx is None:
    cfg = config_dict
    sweep_size = 1

  else:
    keys = list(config_dict.keys())
    values = [val if isinstance(val, list) else [val] for val in config_dict.values()]
    combinations = list(product(*values))

    sweep_size = len(combinations)
    if FLAGS.job_idx >= sweep_size:
      raise ValueError('job_idx exceeds the total number of hyperparam combinations.')

    combination = combinations[FLAGS.job_idx]
    cfg = {keys[i]: combination[i] for i in range(len(keys))}

  return Config(**cfg), sweep_size


def init_wandb(cfg):
  """Initalizes a wandb run"""
  # os.environ['WANDB__SERVICE_WAIT'] = '600'
  # os.environ['WANDB_SILENT'] = 'true'

  if getattr(cfg, 'check_existing_wandb_run', False):
    if _matching_wandb_run_exists(cfg):
      raise FileExistsError('A run with the same config exists. Aborting.')

  name = cfg.wandb_run_name
  if FLAGS.job_idx is not None:
    name = name + f'_job_idx_{FLAGS.job_idx}'

  wandb.init(
    project=cfg.wandb_project,
    mode=cfg.wandb_mode,
    entity=cfg.wandb_entity,
    name=name,
    dir=cfg.wandb_dir,
    config=cfg._asdict(),
  )


def log_job_info():
  """Logs info about cluster job."""
  if FLAGS.job_cluster is not None and FLAGS.job_idx is not None:
    print(f'JOB_CLUSER = {FLAGS.job_cluster}')
    print(f'JOB_INDEX = {FLAGS.job_idx}')
    print(f'JOB_ID = {FLAGS.job_cluster}.{FLAGS.job_idx}')
    wandb.log({'JOB_CLUSTER': FLAGS.job_cluster})
    wandb.log({'JOB_INDEX': FLAGS.job_idx})
    wandb.log(
      {
        'JOB_ID': f'{FLAGS.job_cluster}.{FLAGS.job_idx}',
      }
    )


def _matching_wandb_run_exists(cfg):
  return False


def get_exp_dir_path(cfg):
  """Build a exp_dir path from config. It supports job arrays."""
  exp_dir = os.path.join(cfg.out_dir, cfg.exp_name)
  if FLAGS.job_idx is not None:  # subfolder for each job in the sweep
    exp_dir = os.path.join(exp_dir, f'job_idx_{FLAGS.job_idx}')
  return exp_dir


def maybe_make_dir(cfg):
  """Creates an experiment directory if checkpointing is enabled"""
  if not cfg.save_intermediate_checkpoints and not cfg.save_last_checkpoint:
    return
  if cfg.resume and cfg.resume_exp_name is None:  # if resuming from the same exp
    return

  exp_dir = get_exp_dir_path(cfg)

  if os.path.exists(exp_dir):
    if not cfg.over_write:
      raise ValueError(f'Found existing exp_dir at {exp_dir}.')
    print(f'Removing experiment dir: {exp_dir}')
    shutil.rmtree(exp_dir)

  print(f'Creating experiment directory: {exp_dir}')
  os.makedirs(exp_dir, exist_ok=True)
  with open(os.path.join(exp_dir, 'config.yaml'), 'w') as file:
    yaml.dump(cfg._asdict(), file, default_flow_style=False)


def log(
  cfg, metrics, micro_step, train_loss, train_loss_array, valid_loss, optimizer, world_size, throughput_metrics=None
):
  """Update metrics, print to console, log on wandb."""

  if isinstance(train_loss_array, list):
    train_loss_avg = torch.stack(train_loss_array).mean().item()
  elif isinstance(train_loss_array, torch.Tensor):
    train_loss_avg = train_loss_array.item()

  new_metrics = {
    'micro_step': micro_step,
    'step': micro_step // cfg.grad_accumulation_steps,
    'tokens': micro_step * cfg.micro_batch_size * cfg.seq_len * world_size,
    'lr': optimizer.param_groups[0].get('lr', float('NaN')),
    'train/loss': train_loss.item(),
    'train/loss_avg': train_loss_avg,
    'train/ppl': math.exp(train_loss),
    'train/ppl_avg': math.exp(train_loss_avg),
  }
  if throughput_metrics is not None:
    step_time, flops_per_step = throughput_metrics
    tokens_this_step = cfg.micro_batch_size * cfg.seq_len * cfg.grad_accumulation_steps * world_size
    tokens_per_sec = tokens_this_step / step_time

    gpu_name = torch.cuda.get_device_name(0)
    gpu_peak_flops_per_sec = GPU_PEAK_FLOPS_PER_SEC_MAP.get(gpu_name, None)
    if gpu_peak_flops_per_sec is not None:
      mfu = (flops_per_step / step_time) / world_size * gpu_peak_flops_per_sec

    new_metrics.update(
      {
        'throughput/step_time': step_time,
        'throughput/tokens_per_sec': tokens_per_sec,
        'throughput/mfu': mfu,
        'throughput/flops_per_step': flops_per_step,
      }
    )

  if valid_loss is not None:
    new_metrics['valid/loss'] = valid_loss
    new_metrics['valid/ppl'] = math.exp(valid_loss)

  for k, v in new_metrics.items():
    metrics[k].append(v)

  if cfg.print_progress:
    msg = ' | '.join(
      f'{key}: {value:.3e}' if isinstance(value, float) else f'{key}: {value}' for key, value in new_metrics.items()
    )
    print(msg)

  if cfg.use_wandb:
    wandb.log(new_metrics)


def print_master(msg):
  """Prints only in master process if using multiple GPUs."""
  rank = os.environ.get('RANK', -1)
  ddp = int(rank) != -1
  master_process = (not ddp) or (int(rank) == 0)
  if master_process:
    print(msg)

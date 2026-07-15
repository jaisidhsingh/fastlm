import os
import shutil
import typing as tp
from itertools import product
from types import SimpleNamespace

import torch
import wandb
import yaml
from absl import flags
from huggingface_hub import HfApi
from pandas._libs.parsers import na_values

from src.constants import DEFAULT_CONFIG, SCALING_LADDER, SCALING_RESULTS_FOLDER
from src.utils.throughput_utils import parse_throughput_metrics

FLAGS = flags.FLAGS


def parse_arch_id(arch_id: str) -> tp.Tuple[str, int]:
  """
  - pure attention corresponds to `arch_id = "attn"`
  - pure gdn corresponds to `arch_id = "gdn"`
  - hybrid with gdn:attn = x:1 corresponds to `arch_id: "gdn+attn_x-1"`
  - hybrid with gdn:attn = 1:x corresponds to `arch_id: "gdn+attn_1-x"`
  """
  split_id = arch_id.split('_')
  arch = split_id[0]
  ratio = 1
  if len(split_id) == 2:
    [r1, r2] = [int(x) for x in split_id[1].split('-')]
    if r2 == 1:  # spec follows x:1 format
      ratio = r1
    else:  # spec follows 1:x format
      assert r1 == 1, 'Hybridisation ratio specified incorrectly'
      ratio = int(-1 * r2)  # negative value of ratio handled in `Transformer._prepare_layers`
  return arch, ratio


def set_arch(cfg):
  arch, ratio = parse_arch_id(cfg.arch_id)
  cfg.token_mixer = arch
  cfg.hybrid_mixer_ratio = ratio


def set_eval_every(cfg):
  cfg.eval_every_steps = max(1, int(cfg.steps_budget // cfg.num_evals))


def set_batch_sizes(cfg, world_size, cluster_id):
  peak_mbs = SCALING_LADDER['models'][cfg.param_scale_id]['peak_mbs'][cluster_id]
  req_mbs = int(cfg.global_batch_size // world_size)

  if req_mbs > peak_mbs:
    cfg.micro_batch_size = peak_mbs
  else:
    cfg.micro_batch_size = req_mbs
  cfg.grad_accumulation_steps = int(req_mbs // cfg.micro_batch_size)


def get_steps_from_chinchilla_multiplier(cfg, non_embed_params: int, world_size: int) -> int:
  token_budget = cfg.chinchilla_token_multiplier * non_embed_params * 20
  return int(round(token_budget / (cfg.micro_batch_size * cfg.seq_len * cfg.grad_accumulation_steps * world_size)))


def set_token_budget_id_from_gbs(cfg):
  # if cfg.global_batch_size == 8:
  #   return '1.0B'
  if not cfg.resume:
    mp = SCALING_LADDER['batch_size_vs_token_budget_strategy']['staggered_runs']
    cfg.token_budget_id = mp[cfg.global_batch_size]
  if cfg.resume:
    if cfg.cooldown_only:
      # then we will train until we reach `cfg.token_budget_id` tokens, should already exist in the `.yaml` file
      assert cfg.token_budget_id is not None
    else:
      # then we know that we want to keep training
      mp = SCALING_LADDER['batch_size_vs_token_budget_strategy']['staggered_runs']
      cfg.token_budget_id = mp[cfg.global_batch_size]


def get_steps_from_token_budget_id(cfg, world_size):
  tokens = float(cfg.token_budget_id[:-1]) * 1e9
  return int(round(tokens / (cfg.micro_batch_size * cfg.seq_len * cfg.grad_accumulation_steps * world_size)))


def get_steps_budget(cfg, world_size: int) -> int:
  assert cfg.steps_budget == -1
  if not cfg.resume:
    return get_steps_from_token_budget_id(cfg, world_size)
  else:
    if cfg.cooldown_only:
      return get_steps_from_token_budget_id(cfg, world_size)
    else:
      assert cfg.resume_step is not None
      return get_steps_from_token_budget_id(cfg, world_size)


def load_config_from_constants(param_scale_id):
  config_dict = DEFAULT_CONFIG
  for k in ['d_model', 'n_layers', 'n_heads']:
    config_dict[k] = SCALING_LADDER['models'][param_scale_id][k]
  return SimpleNamespace(**config_dict)


def load_config(path):
  with open(path, 'r') as file:
    config_dict = yaml.safe_load(file)
  # Config = namedtuple('Config', config_dict.keys())

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

  return SimpleNamespace(**cfg), sweep_size


def init_wandb(cfg):
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
    config=vars(cfg),
  )


def log_job_info():
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


def get_exp_dir_path(cfg, world_size, cluster_id):
  gbs = int(cfg.micro_batch_size * cfg.grad_accumulation_steps * world_size)
  gbs_folder = os.path.join(
    SCALING_RESULTS_FOLDER[cluster_id], cfg.arch_id, cfg.param_scale_id, 'gbs_wise_results', f'gbs_{gbs}'
  )
  exp_folder = os.path.join(gbs_folder, 'checkpoints', f'lr_{str(cfg.lr).replace(".", "p")}')
  os.makedirs(exp_folder, exist_ok=True)
  return exp_folder


# def maybe_make_dir(cfg, world_size):
#   if not cfg.save_intermediate_checkpoints and not cfg.save_last_checkpoint:
#     return
#   if cfg.resume and cfg.resume_exp_name is None:  # if resuming from the same exp
#     return

#   exp_dir = get_exp_dir_path(cfg, world_size)

#   if os.path.exists(exp_dir):
#     if not cfg.over_write:
#       raise ValueError(f'Found existing exp_dir at {exp_dir}.')
#     print(f'Removing experiment dir: {exp_dir}')
#     shutil.rmtree(exp_dir)

#   print(f'Creating experiment directory: {exp_dir}')
#   os.makedirs(exp_dir, exist_ok=True)
#   with open(os.path.join(exp_dir, 'config.yaml'), 'w') as file:
#     yaml.dump(cfg._asdict(), file, default_flow_style=False)


def log(
  cfg, metrics, micro_step, train_loss, train_loss_array, valid_loss, optimizer, world_size, throughput_metrics=None
):
  if isinstance(train_loss_array, list):
    train_loss_avg = torch.stack(train_loss_array).mean().item() if train_loss_array else 0.0
  elif isinstance(train_loss_array, torch.Tensor):
    train_loss_avg = train_loss_array.item()
  else:
    train_loss_avg = 0.0

  new_metrics = {
    'micro_step': micro_step,
    'step': micro_step // cfg.grad_accumulation_steps,
    'tokens': micro_step * cfg.micro_batch_size * cfg.seq_len * world_size,
    'lr': optimizer.param_groups[0].get('lr', float('NaN')),
    'train/loss': train_loss.item(),
    'train/loss_avg': train_loss_avg,
  }
  if throughput_metrics is not None:
    new_metrics.update(parse_throughput_metrics(throughput_metrics, cfg, world_size))

  if valid_loss is not None:
    new_metrics['valid/loss'] = valid_loss

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


def download_hf_file(ckpt_params: tuple, dest_folder: str) -> None:
  (arch_id, n, d, gbs, lr) = ckpt_params
  path_in_repo = os.path.join(n, f'gbs_{gbs}', str(lr).replace('.', 'p'), f'ckpt_decayed_to_{d.replace(".", "p")}.pt')
  os.makedirs(dest_folder, exist_ok=True)

  api = HfApi()
  repo_id = 'jaisidhsingh/' + str(arch_id).replace('+', '-')
  save_path = api.hf_hub_download(repo_id=repo_id, filename=repo_in_path, local_dir=dest_folder)
  return save_path


def validate_hf_stored_ckpt(arch_id, n, ckpt):
  err_msg = 'Found incorrect architecture in specified checkpoint!'
  d_model = ckpt['embed_tokens.weight'].shape[1]
  assert d_model == SCALING_LADDER[n]['d_model'], err_msg

  arch, ratio = parse_arch_id(arch_id)

  if '+' not in arch:
    if arch == 'attn':
      assert 'layers.0.token_mixer.w_qkv.weight' in ckpt, err_msg
    else:  # gdn
      assert 'layers.0.token_mixer.A_log' in ckpt, err_msg

  if ratio > 0:
    token_mixers = arch.split('+')  # [gdn, attn]
  else:
    token_mixers = arch.split('+')
    token_mixers.reverse()  # [attn, gdn]

  for i in range(4):
    if ratio > 0:
      if (i + 1) % (ratio + 1) == 0:  # on attn
        assert f'layers.{i}.token_mixer.w_qkv.weight' in ckpt, err_msg
      else:  # on gdn
        assert f'layers.{i}.token_mixer.A_log' in ckpt, err_msg

    else:
      if i % (abs(ratio) + 1) == 0:  # on gdn
        assert f'layers.{i}.token_mixer.A_log' in ckpt, err_msg
      else:  # on attn
        assert f'layers.{i}.token_mixer.w_qkv.weight' in ckpt, err_msg

  print('Checkpoint architecture is valid!')

import json
import os
import subprocess
import typing as tp
from copy import deepcopy
from dataclasses import dataclass

import pandas as pd
import tyro

from src.constants import DEFAULT_CONFIG, SCALING_LADDER

LR_FLOAT_TO_STR_MAP = {0.00025: '25e-5', 0.0005: '5e-4', 0.001: '1e-3', 0.002: '2e-3', 0.004: '4e-3', 0.008: '8e-3'}
PARAM_SCALE_ID_TO_MEM_MAP = {'20M': 64, '50M': 72, '150M': 96, '300M': 128}
DB_PATH = '/home/jsingh/projects/fastlm/execs/exec_db.csv'
FOLDER = '/home/jsingh/projects/fastlm/execs'


@dataclass
class MainConfigJob:
  bid: int
  arch_id: str
  n: str
  gbs: int
  lr: tp.Union[str, float]
  mode: str


def check_subfolders(cfg):
  os.makedirs(os.path.join(FOLDER, cfg.arch_id), exist_ok=True)
  os.makedirs(os.path.join(FOLDER, cfg.arch_id, cfg.n), exist_ok=True)


def get_dp_value(n, gbs):
  if gbs in [16, 32]:
    return 1
  elif gbs in [64, 128]:
    if n in ['20M', '50M']:
      return 2
    else:  # n in ["150M", "300M"]
      return 4
  else:  # gbs in [256, 512]
    if n in ['20M', '50M']:
      return 4
    else:  # n in ["150M", "300M"]
      return 8


def get_config_path(arch_id, n, gbs, lr, mode):
  lr_ext = 'all_parallel' if isinstance(lr, list) else LR_FLOAT_TO_STR_MAP[lr]
  return f'/home/jsingh/projects/fastlm/execs/{arch_id}/{n}/cfg-{mode}={gbs}_lr={lr_ext}.yaml'


def get_jobfile_path(arch_id, n, gbs, lr, mode):
  lr_ext = 'all_parallel' if isinstance(lr, list) else LR_FLOAT_TO_STR_MAP[lr]
  return f'/home/jsingh/projects/fastlm/execs/{arch_id}/{n}/job-{mode}_gbs={gbs}_lr={lr_ext}.sub'


def get_config_content(arch_id, n, gbs, lr, mode):
  base_cfg = deepcopy(DEFAULT_CONFIG)

  # model specifications first
  base_cfg['arch_id'] = arch_id
  base_cfg['param_scale_id'] = n

  for k in ['d_model', 'n_layers', 'n_heads']:
    base_cfg[k] = SCALING_LADDER['models'][n][k]

  # batch size
  base_cfg['global_batch_size'] = gbs
  base_cfg['micro_batch_size'] = -1
  base_cfg['grad_accumulation_steps'] = 1

  # token budget
  base_cfg['token_budget_id'] = SCALING_LADDER['batch_size_vs_token_budget_strategy']['staggered_grid'][gbs]
  base_cfg['steps_budget'] = -1

  # learning rate
  base_cfg['lr'] = lr
  if n in ['150M', '300M']:
    base_cfg['beta2'] = 0.95

  # TODO: resume -> learning rate decay
  if cfg.mode == 'decay':
    pass

  return base_cfg


def get_jobfile_content(arch_id, n, gbs, lr, mode, cpus=8):
  dp = get_dp_value(n, gbs)
  single_or_multi = 'single' if dp == 1 else 'multi'
  n_jobs = len(lr) if isinstance(lr, list) else 1
  mem = PARAM_SCALE_ID_TO_MEM_MAP[n]

  return f"""
  # Executable should be a full path
  executable=/home/jsingh/projects/fastlm/cluster/{single_or_multi}_gpu/condor.sh

  # Hyperparmeters are specified in a YAML configuration file
  # config={get_config_path(arch_id, n, gbs, lr, mode)}

  # Queue as many jobs as points in the hyperaparameter grid
  n_jobs={n_jobs}

  # Pass arguments to the executable
  arguments = $(config) $(Process) $(Cluster)

  # Logs
  LOGS_DIR=/fast/jsingh/logs/fastlm/june/attn

  error = $(LOGS_DIR)/err/job.$(Cluster).$(Process).err
  output = $(LOGS_DIR)/out/job.$(Cluster).$(Process).out
  log = $(LOGS_DIR)/log/job.$(Cluster).$(Process).log

  # Job requirements
  request_memory = {mem}G
  request_cpus = {cpus}
  request_gpus = {dp}
  requirements = (TARGET.CUDADeviceName == "NVIDIA A100-SXM4-80GB" || TARGET.CUDADeviceName == "NVIDIA H100 80GB HBM3")

  queue $(n_jobs)
  """


def main(cfg):
  check_subfolders(cfg)

  lr = None
  if isinstance(cfg.lr, str):
    if cfg.lr == 'all_parallel':
      lr = SCALING_LADDER['learning_rates']
    else:
      raise NotImplementedError('No other string options supported for `cfg.lr`')
  else:
    assert isinstance(cfg.lr, float), 'Learning rate must either be a string `all_parallel` or a float.'
    lr = cfg.lr

  assert lr is not None

  # first create and save the config
  config_path = get_config_path(cfg.arch_id, cfg.n, cfg.gbs, lr, cfg.mode)
  config = get_config_content(cfg.arch_id, cfg.n, cfg.gbs, lr, cfg.mode)
  with open(config_path, 'w') as f:
    json.dump(config, f)

  # then use this config path in the submission file
  jobfile_path = get_jobfile_path(cfg.arch_id, cfg.n, cfg.abs, lr, cfg.mode)
  jobfile_content = get_jobfile_content(cfg.arch_id, cfg.n, cfg.gbs, lr, cfg.mode)
  with open(jobfile_path, 'w') as f:
    f.write(jobfile_content)

  # submit this job
  result = subprocess.run(
    ['condor_submit_bid', f'{cfg.bid}', jobfile_path],  # command and arguments
    capture_output=True,  # capture stdout/stderr
    text=True,  # return strings instead of bytes
  )
  if result.returncode == 0:
    cluster_id = result.stdout.split(' submitted to cluster ')[-1].replace('.', '')
    cluster_id = int(cluster_id)

    # update our file that contains info about what we ran
    info = {
      'arch_id': cfg.arch_id,
      'n': cfg.n,
      'gbs': cfg.gbs,
      'lr': 'all_parallel' if isinstance(lr, list) else lr,
      'dp': get_dp_value(cfg.n, cfg.gbs),
      'main': 'yes' if cfg.mode == 'main' else 'no',
      'decay': 'yes' if cfg.mode == 'decay' else 'no',
      'cfg': 'yes',
      'sub': 'yes',
      'cluster_id': cluster_id,
      'n_jobs': len(lr) if isinstance(lr, list) else 1,
    }
    new_row = pd.DataFrame([info])

    try:
      df = pd.read_csv(DB_PATH)
    except (FileNotFoundError, pd.errors.EmptyDataError):
      df = pd.DataFrame()

    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(DB_PATH, index=False)

  else:
    print('Something bad happened when we submit the job using subprocess! Printing the subprocess call error:')
    print(result.stderr)


if __name__ == '__main__':
  cfg = tyro.cli(MainConfigJob)
  print('main launching')
  # main(cfg)

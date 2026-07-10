import os
import typing as tp
from dataclasses import dataclass

import tyro
import yaml

from manager.manager_utils import (
  get_cluster_prefix,
  get_config_path,
  get_eval_config_content,
  get_eval_jobfile_content,
  get_jobfile_path,
  get_train_config_content,
  get_train_jobfile_content,
  parse_input_lr,
  submit_and_log,
)


@dataclass
class ManagerConfig:
  bid: int = 1
  arch_id: str = 'attn'
  n: str = '20M'
  gbs: int = 32
  lr: tp.Union[str, float] = 'all_parallel'
  mode: str = 'main'
  submit: str = 'yes'
  routine: str = 'train'
  cluster_id: str = 'mpi'
  # if routine == "eval"
  benchmarks: str = 'ruler,dclm_core'  # comma-separated benchmark list
  ckpt_token_budget: str = '3.0B'


def check_subfolders(cfg: ManagerConfig):
  folder = os.path.join(get_cluster_prefix(cfg.cluster_id), 'execs')
  os.makedirs(os.path.join(folder, cfg.arch_id), exist_ok=True)
  os.makedirs(os.path.join(folder, cfg.arch_id, cfg.n), exist_ok=True)


def train_management(cfg: ManagerConfig):
  check_subfolders(cfg)
  lr = parse_input_lr(cfg)

  # first create and save the config
  config_path = get_config_path(cfg, lr)
  config = get_train_config_content(cfg, lr)
  with open(config_path, 'w') as f:
    yaml.safe_dump(config, f)

  # count the number of parallel jobs to launch under the same cluster id
  n_jobs = 1
  for k, v in config.items():
    if isinstance(v, list):
      n_jobs *= len(v)
  n_jobs = int(n_jobs)

  # then use this config path in the submission file
  jobfile_path = get_jobfile_path(cfg, lr)
  jobfile_content = get_train_jobfile_content(cfg, lr, n_jobs, cpus=config['num_workers'])
  with open(jobfile_path, 'w') as f:
    f.write(jobfile_content)

  print(f'Wrote config -> {config_path}')
  print(f'Wrote job    -> {jobfile_path}  (n_jobs={n_jobs})')

  if cfg.submit == 'yes':
    # set up the batch-job command according to the cluster we're on
    if cfg.cluster_id == 'mpi':
      cmdlist = ['condor_submit_bid', f'{cfg.bid}', jobfile_path]
    elif cfg.cluster_id in ['capella', 'alpha']:
      cmdlist = ['sbatch', jobfile_path]
    else:
      raise ValueError('Unsupported value found for `--cluster_id`.')

    # submit the batch-job and store it
    submit_and_log(cmdlist, cfg, jobfile_path, lr, n_jobs)


def eval_management(cfg: ManagerConfig):
  check_subfolders(cfg)
  benchmarks = [b.strip() for b in cfg.benchmarks.split(',') if b.strip()]

  # parse learning rate(s) such that each lr gets its own yaml + job file
  # because checkpoint paths depend on lr and can't be cross-product'ed
  lrs = parse_input_lr(cfg)
  if not isinstance(lrs, list):
    lrs = [lrs]

  for lr in lrs:
    # build and save the eval config
    config_path = get_config_path(cfg, lr)
    config = get_eval_config_content(cfg, lr, benchmarks)
    with open(config_path, 'w') as f:
      yaml.safe_dump(config, f, default_flow_style=False)

    # count n_jobs: only `benchmarks` is a list in the config
    n_jobs = len(benchmarks)

    # build and save the job file
    jobfile_path = get_jobfile_path(cfg, lr)
    jobfile_content = get_eval_jobfile_content(cfg, lr, n_jobs, cpus=config['num_workers'])
    with open(jobfile_path, 'w') as f:
      f.write(jobfile_content)

    print(f'Wrote config -> {config_path}')
    print(f'Wrote job    -> {jobfile_path}  (n_jobs={n_jobs})')

    if cfg.submit == 'yes':
      if cfg.cluster_id == 'mpi':
        cmdlist = ['condor_submit_bid', f'{cfg.bid}', jobfile_path]
      elif cfg.cluster_id in ['capella', 'alpha']:
        cmdlist = ['sbatch', jobfile_path]
      else:
        raise ValueError('Unsupported value found for `--cluster_id`.')

      # submit the batch-job and store it
      submit_and_log(cmdlist, cfg, jobfile_path, lr, n_jobs)


def main(cfg: ManagerConfig):
  if cfg.routine == 'train':
    train_management(cfg)
  elif cfg.routine == 'eval':
    eval_management(cfg)
  else:
    raise NotImplementedError(
      'Supported values for `cfg.routine` are `[train, eval]`. You provided an unsupported value.'
    )


if __name__ == '__main__':
  cfg = tyro.cli(ManagerConfig)
  main(cfg)

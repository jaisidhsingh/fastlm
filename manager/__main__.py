import os
import subprocess
import typing as tp
from copy import deepcopy
from dataclasses import dataclass

import pandas as pd
import tyro
import yaml

from src.constants import DEFAULT_CONFIG, SCALING_LADDER

LR_FLOAT_TO_STR_MAP = {0.00025: '25e-5', 0.0005: '5e-4', 0.001: '1e-3', 0.002: '2e-3', 0.004: '4e-3', 0.008: '8e-3'}
PARAM_SCALE_ID_TO_MEM_MAP = {'20M': 64, '50M': 72, '150M': 96, '300M': 128}
DB_PATH = '/home/jsingh/projects/fastlm/execs/exec_db.csv'
FOLDER = '/home/jsingh/projects/fastlm/execs'


@dataclass
class ManagerConfig:
  bid: int = 1
  arch_id: str = 'attn'
  n: str = '20M'
  gbs: int = 32
  lr: tp.Union[str, float] = 'all_parallel'
  mode: str = 'main'


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
  return f'/lustre/home/jsingh/projects/fastlm/execs/{arch_id}/{n}/cfg-{mode}_gbs-{gbs}_lr-{lr_ext}.yaml'


def get_jobfile_path(arch_id, n, gbs, lr, mode):
  lr_ext = 'all_parallel' if isinstance(lr, list) else LR_FLOAT_TO_STR_MAP[lr]
  return f'/lustre/home/jsingh/projects/fastlm/execs/{arch_id}/{n}/job-{mode}_gbs-{gbs}_lr-{lr_ext}.sub'


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
  base_cfg['token_budget_id'] = SCALING_LADDER['batch_size_vs_token_budget_strategy']['staggered_runs'][gbs]
  base_cfg['steps_budget'] = -1

  # learning rate
  base_cfg['lr'] = lr
  if n in ['150M', '300M']:
    base_cfg['beta2'] = 0.95

  # saving
  base_cfg['save_last_checkpoint'] = True
  base_cfg['save_intermediate_checkpoints'] = True

  # wandb
  base_cfg['use_wandb'] = True
  base_cfg['wandb_mode'] = 'offline'
  base_cfg['wandb_run_name'] = f'{arch_id}-{n}_gbs-{gbs}'
  base_cfg['exp_name'] = f'{arch_id}-{n}_gbs-{gbs}'

  # resume -> learning rate decay (cooldown only)
  if mode == 'decay':
    full_token_budget_id = deepcopy(base_cfg['token_budget_id'])
    token_budgets = list(SCALING_LADDER['batch_size_vs_token_budget_strategy']['staggered_grid'].keys())
    full_budget_index = token_budgets.index(full_token_budget_id)
    budgets_to_decay_at = token_budgets[:full_budget_index]

    # we set `token_budget_id` key to a `list` type so that
    # `utils.load_config` can split the list into individual runs
    # and submit decays jobs in parallel. `checkpoint_utils` will use
    # `token_budget_id` to load in the correct checkpoint
    base_cfg['token_budget_id'] = budgets_to_decay_at
    base_cfg['resume'] = True
    base_cfg['resume_step'] = None
    base_cfg['resume_exp_name'] = f'decay_starts_to_{full_token_budget_id.replace(".", "p")}'
    base_cfg['cooldown_only'] = True
    base_cfg['save_last_checkpoint'] = False
    base_cfg['use_wandb'] = False
    base_cfg['wandb_run_name'] = f'{arch_id}-{n}_gbs-{gbs}_cooldown'
    base_cfg['exp_name'] = f'{arch_id}-{n}_gbs-{gbs}_cooldown'

  return base_cfg


def get_jobfile_content(arch_id, n, gbs, lr, n_jobs, mode, cpus=8):
  dp = get_dp_value(n, gbs)
  single_or_multi = 'single' if dp == 1 else 'multi'
  mem = PARAM_SCALE_ID_TO_MEM_MAP[n]

  args = '$(config) $(Process) $(Cluster)'
  if dp > 1:
    args = args + ' $(dp)'

  return f"""# Executable should be a full path
executable=/home/jsingh/projects/fastlm/cluster/{single_or_multi}_gpu/condor.sh

# Hyperparmeters are specified in a YAML configuration file
config={get_config_path(arch_id, n, gbs, lr, mode)}

# Queue as many jobs as points in the hyperaparameter grid
n_jobs={n_jobs}
dp={dp}

# Pass arguments to the executable
arguments = {args}

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


def sanity_check():
  cfg = ManagerConfig(n='20M', arch_id='attn', gbs=32, lr='all_parallel', bid=1, mode='sanity_check')
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

  refs = {
    'main': lambda x: (
      '/lustre/home/jsingh/projects/fastlm/june_exec/N-20M,50M_gbs-16,32/cfg_20M_gbs-32_all-lr_parallel.yaml'
    ),
    'decay': lambda x: (
      f'/lustre/home/jsingh/projects/fastlm/june_exec/decay_intermediates/cfg_20M_gbs-32_tb-{x.replace(".", "p")}.yaml'
    ),
  }

  for mode in ['main', 'decay']:
    cfg.mode = mode
    config = get_config_content(cfg.arch_id, cfg.n, cfg.gbs, lr, cfg.mode)

    if mode == 'main':
      with open(refs['main'](None)) as f:
        ref_config = dict(yaml.safe_load(f))

      for k in config.keys():
        assert k in ref_config and ref_config[k] == config[k], (
          f'Key={k} erroneous: ref={ref_config[k]}, cfg={config[k]}'
        )

    else:
      for tbid in config['token_budget_id']:
        tmp_config = deepcopy(config)
        tmp_config['token_budget_id'] = tbid
        tmp_config['resume_exp_name'] = f'decay_starts_to_{tbid.replace(".", "p")}'
        with open(refs['decay'](tbid)) as f:
          ref_config = dict(yaml.safe_load(f))

        for k in tmp_config.keys():
          assert k in ref_config and ref_config[k] == tmp_config[k], (
            f'Token budget id={tbid}, Key={k} erroneous, ref={ref_config[k]}, cfg={tmp_config[k]}'
          )


def main(cfg: ManagerConfig):
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
    yaml.safe_dump(config, f)

  # count the number of parallel jobs to launch under the same cluster id
  n_jobs = 1
  for k, v in config.items():
    if isinstance(v, list):
      n_jobs *= len(v)
  n_jobs = int(n_jobs)

  # then use this config path in the submission file
  jobfile_path = get_jobfile_path(cfg.arch_id, cfg.n, cfg.gbs, lr, cfg.mode)
  jobfile_content = get_jobfile_content(cfg.arch_id, cfg.n, cfg.gbs, lr, n_jobs, cfg.mode)
  with open(jobfile_path, 'w') as f:
    f.write(jobfile_content)

  # submit this job
  result = subprocess.run(
    ['condor_submit_bid', f'{cfg.bid}', jobfile_path],  # command and arguments
    capture_output=True,  # capture stdout/stderr
    text=True,  # return strings instead of bytes
  )
  if result.returncode == 0:
    print(result.stdout)
    cluster_id = result.stdout.split(' cluster ')[-1][:-2]
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
      'n_jobs': n_jobs,
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
  # sanity_check()
  cfg = tyro.cli(ManagerConfig)
  main(cfg)

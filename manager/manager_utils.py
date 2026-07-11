import os
import subprocess
import typing as tp
from copy import deepcopy

import pandas as pd
import tyro

from src.constants import (
  DATA_PREFIXES,
  DEFAULT_CONFIG,
  LR_FLOAT_TO_STR_MAP,
  PARAM_SCALE_ID_TO_MEM_MAP,
  SCALING_LADDER,
  SCALING_RESULTS_FOLDER,
  TOKENIZERS,
  WANDB_DIR_PREFIXES,
)


def parse_arch_id(arch_id: str) -> tp.Tuple[str, int]:
  split_id = arch_id.split('_')
  arch = split_id[0]
  ratio = 1
  if len(split_id) == 2:
    ratio = int(split_id[1].split('-')[0])
  return arch, ratio


def parse_input_lr(cfg):
  lr = None
  if isinstance(cfg.lr, str):
    if cfg.lr == 'all_parallel':
      lr = SCALING_LADDER['learning_rates']
    elif ',' in cfg.lr:
      lr = []
      for x in cfg.lr.split(','):
        if len(x) > 0:
          lr.append(float(x))
    else:
      raise NotImplementedError('No other string options supported for `cfg.lr`')
  else:
    assert isinstance(cfg.lr, float), 'Learning rate must either be a string `all_parallel` or a float.'
    lr = cfg.lr

  assert lr is not None
  return lr


def get_dp_value(cfg):
  gbs = cfg.gbs
  n = cfg.n
  if cfg.cluster_id in ['mpi', 'capella', 'alpha']:
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
  else:
    raise ValueError(f'Found unsupported value: {cfg.cluster_id} of `--cluster_id`')


def get_cluster_prefix(cluster_id):
  if cluster_id == 'mpi':
    return '/lustre/home/jsingh/projects'
  elif cluster_id == 'alpha' or cluster_id == 'capella':
    return '/projects/p_neurasearch'
  else:
    raise ValueError('Unsupported argument `cluster_id`.')


def get_cluster_data_prefix(cluster_id):
  if cluster_id == 'mpi':
    return '/fast/jsingh/data'
  elif cluster_id == 'alpha' or cluster_id == 'capella':
    return '/data/horse/ws/jasi149i-fastlm/data'
  else:
    raise ValueError('Unsupported argument `cluster_id`.')


def get_lr_ext(lr):
  lr_ext = None
  if isinstance(lr, list) and len(lr) == len(list(LR_FLOAT_TO_STR_MAP.keys())):
    lr_ext = 'all_parallel'
  elif isinstance(lr, list):
    lr_ext = '--'.join([LR_FLOAT_TO_STR_MAP[k] for k in lr])
  elif isinstance(lr, float):
    lr_ext = LR_FLOAT_TO_STR_MAP[lr]
  else:
    raise ValueError('Unsupported type or value of `lr`')
  return lr_ext


def get_jobfile_path(cfg, lr):
  lr_ext = get_lr_ext(lr)
  return f'{get_cluster_prefix(cfg.cluster_id)}/fastlm/execs/{cfg.arch_id}/{cfg.n}/{cfg.routine}-job-{cfg.mode}_gbs-{cfg.gbs}_lr-{lr_ext}.sub'


def get_config_path(cfg, lr):
  lr_ext = get_lr_ext(lr)
  return f'{get_cluster_prefix(cfg.cluster_id)}/fastlm/execs/{cfg.arch_id}/{cfg.n}/{cfg.routine}-cfg-{cfg.mode}_gbs-{cfg.gbs}_lr-{lr_ext}.yaml'


def get_train_config_content(cfg, lr):
  base_cfg = deepcopy(DEFAULT_CONFIG)

  # model specifications first
  base_cfg['arch_id'] = cfg.arch_id
  base_cfg['param_scale_id'] = cfg.n
  mixer, ratio = parse_arch_id(cfg.arch_id)
  base_cfg['token_mixer'] = mixer
  base_cfg['hybrid_mixer_ratio'] = ratio

  for k in ['d_model', 'n_layers', 'n_heads']:
    base_cfg[k] = SCALING_LADDER['models'][cfg.n][k]

  # data
  base_cfg['trainset_path'] = os.path.join(DATA_PREFIXES[cfg.cluster_id], base_cfg['trainset_path'])
  base_cfg['validset_path'] = os.path.join(DATA_PREFIXES[cfg.cluster_id], base_cfg['validset_path'])
  base_cfg['num_workers'] = 16 if cfg.cluster_id == 'mpi' else 8

  # batch size
  base_cfg['global_batch_size'] = cfg.gbs
  base_cfg['micro_batch_size'] = -1
  base_cfg['grad_accumulation_steps'] = 1

  # token budget
  base_cfg['token_budget_id'] = SCALING_LADDER['batch_size_vs_token_budget_strategy']['staggered_runs'][cfg.gbs]
  base_cfg['steps_budget'] = -1

  # learning rate
  base_cfg['lr'] = lr
  if cfg.n in ['150M', '300M']:
    base_cfg['beta2'] = 0.95
  base_cfg['warmup_steps'] = 2000
  base_cfg['cooldown_steps'] = 0.2

  # saving
  base_cfg['save_last_checkpoint'] = True
  base_cfg['save_intermediate_checkpoints'] = True

  # wandb
  base_cfg['use_wandb'] = True
  base_cfg['wandb_mode'] = 'offline'
  base_cfg['wandb_dir'] = WANDB_DIR_PREFIXES[cfg.cluster_id]
  base_cfg['wandb_run_name'] = f'{cfg.arch_id}-{cfg.n}_gbs-{cfg.gbs}'
  base_cfg['exp_name'] = f'{cfg.arch_id}-{cfg.n}_gbs-{cfg.gbs}'

  # resume -> learning rate decay (cooldown only)
  if cfg.mode == 'decay':
    full_token_budget_id = deepcopy(base_cfg['token_budget_id'])
    token_budgets = list(SCALING_LADDER['batch_size_vs_token_budget_strategy']['staggered_grid'].keys())
    full_budget_index = token_budgets.index(full_token_budget_id)
    budgets_to_decay_at = token_budgets[:full_budget_index]

    # we set `token_budget_id` key to a `list` type so that
    # `utils.load_config` can split the list into individual runs
    # and submit decays jobs in parallel. `checkpoint_utils` will use
    # `token_budget_id` to load in the correct checkpoint
    base_cfg['token_budget_id'] = budgets_to_decay_at
    base_cfg['scheduler'] = 'linear_cooldown'
    base_cfg['resume'] = True
    base_cfg['resume_step'] = None
    base_cfg['resume_exp_name'] = f'decay_starts_to_{full_token_budget_id.replace(".", "p")}'
    base_cfg['cooldown_only'] = True
    base_cfg['save_last_checkpoint'] = False
    base_cfg['use_wandb'] = False
    base_cfg['wandb_run_name'] = f'{cfg.arch_id}-{cfg.n}_gbs-{cfg.gbs}_cooldown'
    base_cfg['exp_name'] = f'{cfg.arch_id}-{cfg.n}_gbs-{cfg.gbs}_cooldown'

  return base_cfg


def get_eval_config_content(cfg, lr, benchmarks):
  mixer, ratio = parse_arch_id(cfg.arch_id)

  lr_str = str(lr).replace('.', 'p')
  raw_ckpt = os.path.join(
    SCALING_RESULTS_FOLDER['cfg.cluster_id'],
    cfg.arch_id,
    cfg.n,
    'gbs_wise_results',
    f'gbs_{cfg.gbs}',
    'checkpoints',
    f'lr_{lr_str}',
    f'ckpt_{cfg.ckpt_token_budget.replace(".", "p")}.pt',
  )

  return {
    'benchmarks': benchmarks,
    'raw_ckpt': raw_ckpt,
    'tokenizer_path': TOKENIZERS[cfg.cluster_id],
    # Model arch (must match training)
    'arch_id': cfg.arch_id,
    'param_scale_id': cfg.n,
    'model': 'transformer',
    'd_model': SCALING_LADDER['models'][cfg.n]['d_model'],
    'mlp_class': 'glu',
    'expand': '3.0',
    'n_layers': SCALING_LADDER['models'][cfg.n]['n_layers'],
    'n_heads': SCALING_LADDER['models'][cfg.n]['n_heads'],
    'rms_norm': True,
    'tie_embeddings': True,
    'torch_compile': False,
    'use_flex_attention': False,
    'token_mixer': mixer,
    'hybrid_mixer_ratio': ratio,
    'layer_norm_scaling': False,
    'residual_connection': 'add',
    'attn_gate': True,
    'attn_qk_norm': True,
    'gdn_conv_size': 4,
    'gdn_gate': True,
    'gdn_neg_eigval': True,
    'intra_doc_masking': True,
    # Data
    'vocab_size': 50304,
    'seq_len': 2048,
    'dtype': 'bfloat16',
    # Used for results folder path
    'global_batch_size': cfg.gbs,
    'lr': lr,
  }


def get_train_jobfile_content(cfg, lr, n_jobs, cpus=8):
  dp = get_dp_value(cfg)
  single_or_multi = 'single' if dp == 1 else 'multi'
  mem = PARAM_SCALE_ID_TO_MEM_MAP[cfg.n]
  n_jobs_minus_1 = n_jobs - 1

  args = '$(config) $(Process) $(Cluster)'
  if dp > 1:
    args = args + ' $(dp)'

  if cfg.cluster_id == 'mpi':
    return f"""# Executable should be a full path
executable=/home/jsingh/projects/fastlm/cluster/{single_or_multi}_gpu/condor.sh

# Hyperparmeters are specified in a YAML configuration file
config={get_config_path(cfg, lr)}

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
requirements = (TARGET.CUDADeviceName == "NVIDIA A100-SXM4-80GB" || TARGET.CUDADeviceName == "NVIDIA H100 80GB HBM3" || TARGET.CUDADeviceName == "NVIDIA H100") && (Machine != "g174.internal.cluster.is.localnet")

queue $(n_jobs)
  """

  elif cfg.cluster_id == 'alpha':
    return f"""#!/bin/bash
#SBATCH --job-name=somejobname
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}G
#SBATCH --gres=gpu:{dp}
#SBATCH --time=24:00:00
#SBATCH --account=p_neurasearch
#SBATCH --job-name=somejob
#SBATCH --output=/data/horse/ws/jasi149i-fastlm/logs/june/out/job-%A_%a.out
#SBATCH --error=/data/horse/ws/jasi149i-fastlm/logs/june/err/job-%A_%a.err
#SBATCH --array=0-{n_jobs_minus_1}
#SBATCH --exclude=i8009

CONFIG={get_config_path(cfg, lr)}
DP={dp}
HOST=$(hostname -f)

cd /projects/p_neurasearch/fastlm

if [ "$DP" -eq 1 ]; then
    bash cluster/single_gpu/alpha.sh "$CONFIG" "$SLURM_ARRAY_TASK_ID" "$SLURM_JOB_ID"
else
  if [ "$HOST" == *alpha* && "$DP" -eq 16]; then
    bash cluster/multi_gpu/alpha_multinode.sh "$CONFIG" "$SLURM_ARRAY_TASK_ID" "$SLURM_JOB_ID" "$DP"
  else
    bash cluster/multi_gpu/alpha.sh "$CONFIG" "$SLURM_ARRAY_TASK_ID" "$SLURM_JOB_ID" "$DP"
  fi
fi
"""

  elif cfg.cluster_id == 'capella':
    return f"""#!/bin/bash
#SBATCH --job-name=fastlm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}G
#SBATCH --gres=gpu:{dp}
#SBATCH --time=24:00:00
#SBATCH --account=p_neurasearch
#SBATCH --job-name=somejobname
#SBATCH --output=/data/horse/ws/jasi149i-fastlm/logs/june/out/job-%A_%a.out
#SBATCH --error=/data/horse/ws/jasi149i-fastlm/logs/june/err/job-%A_%a.err
#SBATCH --array=0-{n_jobs_minus_1}

CONFIG={get_config_path(cfg, lr)}
DP={dp}
HOST=$(hostname -f)

cd /projects/p_neurasearch/fastlm

if [ "$DP" -eq 1 ]; then
    bash cluster/single_gpu/capella.sh "$CONFIG" "$SLURM_ARRAY_TASK_ID" "$SLURM_JOB_ID"
else
  if [ "$HOST" == *capella* && "$DP" -eq 8]; then
    bash cluster/multi_gpu/capella_multinode.sh "$CONFIG" "$SLURM_ARRAY_TASK_ID" "$SLURM_JOB_ID" "$DP"
  else
    bash cluster/multi_gpu/capella.sh "$CONFIG" "$SLURM_ARRAY_TASK_ID" "$SLURM_JOB_ID" "$DP"
  fi
fi
"""
  else:
    raise ValueError('Unsupported value of cli-arg `--cluster_id` provided')


def get_eval_jobfile_content(cfg, lr, n_jobs, cpus=8):
  mem = PARAM_SCALE_ID_TO_MEM_MAP[cfg.n]
  n_jobs_minus_1 = n_jobs - 1
  if cfg.cluster_id == 'mpi':
    return f"""# Executable should be a full path
executable=/home/jsingh/projects/fastlm/cluster/single_gpu/eval_condor.sh

# Hyperparmeters are specified in a YAML configuration file
config={get_config_path(cfg, lr)}

# Queue as many jobs as points in the hyperaparameter grid (one per benchmark)
n_jobs={n_jobs}
dp=1

# Pass arguments to the executable
arguments = $(config) $(Process) $(Cluster)

# Logs
LOGS_DIR=/fast/jsingh/logs/fastlm/june/eval

error = $(LOGS_DIR)/err/job.$(Cluster).$(Process).err
output = $(LOGS_DIR)/out/job.$(Cluster).$(Process).out
log = $(LOGS_DIR)/log/job.$(Cluster).$(Process).log

# Job requirements
request_memory = {mem}G
request_cpus = {cpus}
request_gpus = 1
requirements = (TARGET.CUDADeviceName == "NVIDIA A100-SXM4-80GB" || TARGET.CUDADeviceName == "NVIDIA H100 80GB HBM3" || TARGET.CUDADeviceName == "NVIDIA H100") && (Machine != "g174.internal.cluster.is.localnet")

queue $(n_jobs)
    """
  elif cfg.cluster_id == 'alpha':
    return f"""#!/bin/bash
#SBATCH --job-name=fastlm_eval
#SBATCH --output=/fast/jsingh/logs/fastlm/june/eval/out/job.%A.%a.out
#SBATCH --error=/fast/jsingh/logs/fastlm/june/eval/err/job.%A.%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100-80gb|h100"
#SBATCH --exclude=g174.internal.cluster.is.localnet
#SBATCH --array=0-{n_jobs_minus_1}

EXECUTABLE=/home/jsingh/projects/fastlm/cluster/single_gpu/eval_alpha.sh

CONFIG={get_config_path(cfg, lr)}

LOGS_DIR=/fast/jsingh/logs/fastlm/june/eval
mkdir -p "$LOGS_DIR/err" "$LOGS_DIR/out" "$LOGS_DIR/log"

srun "$EXECUTABLE" "$CONFIG" "$SLURM_ARRAY_TASK_ID" "$SLURM_JOB_ID"
"""

  elif cfg.cluster_id == 'capella':
    return f"""#!/bin/bash
#SBATCH --job-name=fastlm_eval
#SBATCH --output=/fast/jsingh/logs/fastlm/june/eval/out/job.%A.%a.out
#SBATCH --error=/fast/jsingh/logs/fastlm/june/eval/err/job.%A.%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100-80gb|h100"
#SBATCH --exclude=g174.internal.cluster.is.localnet
#SBATCH --array=0-{n_jobs_minus_1}

EXECUTABLE=/home/jsingh/projects/fastlm/cluster/single_gpu/eval_capella.sh

CONFIG={get_config_path(cfg, lr)}

LOGS_DIR=/fast/jsingh/logs/fastlm/june/eval
mkdir -p "$LOGS_DIR/err" "$LOGS_DIR/out" "$LOGS_DIR/log"

srun "$EXECUTABLE" "$CONFIG" "$SLURM_ARRAY_TASK_ID" "$SLURM_JOB_ID"
"""
  else:
    raise ValueError('Unsupported value of cli-arg `--cluster_id` provided')


def submit_and_log(cmdlist, cfg, jobfile_path, lr, n_jobs):
  result = subprocess.run(
    cmdlist,
    capture_output=True,
    text=True,
  )
  if result.returncode == 0:
    print(result.stdout)
    query_str = ' cluster ' if cfg.cluster_id == 'mpi' else ' batch job '
    cluster_sub_id = result.stdout.split(query_str)[-1][:-2]
    cluster_sub_id = int(cluster_sub_id)

    info = {
      'arch_id': cfg.arch_id,
      'n': cfg.n,
      'gbs': cfg.gbs,
      'lr': lr,
      'dp': get_dp_value(cfg),
      'main': 'yes' if cfg.mode == 'main' else 'no',
      'decay': 'yes' if cfg.mode == 'decay' else 'no',
      'routine': cfg.routine,
      'cfg': 'yes',
      'sub': 'yes',
      'cluster_id': cluster_sub_id,
      'cluster_name': cfg.cluster_id,
      'n_jobs': n_jobs,
    }
    new_row = pd.DataFrame([info])

    db_path = os.path.join(get_cluster_prefix(cfg.cluster_id), 'execs', 'exec_db.csv')
    try:
      df = pd.read_csv(db_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
      df = pd.DataFrame()

    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(db_path, index=False)

  else:
    print('Something bad happened when we submit the job using subprocess! Printing the subprocess call error:')
    print(result.stderr)

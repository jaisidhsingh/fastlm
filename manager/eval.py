import subprocess
import typing as tp
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
import os
import tyro
import yaml
from beartype import beartype

from src.constants import *

D_VALUES = ['0.5B', '1.0B', '3.0B', '7.5B', '15.0B']
DMAP = {
  16: '1.0B',
  32: '3.0B',
  64: '15.0B',
  128: '15.0B',
  256: '15.0B',
  512: '15.0B',
}


@beartype
@dataclass
class Config:
  arch_id: str
  n: str
  d: str
  gbs: int
  lr: float
  bid: int
  submit: str
  cluster_id: str


def parse_cfg(input_cfg):
  cfg = deepcopy(input_cfg)
  if input_cfg.d == 'all_parallel':
    d_index = D_VALUES.index(DMAP[input_cfg.gbs])
    cfg.d = ','.join(D_VALUES[:d_index+1])
  return cfg


def get_cluster_prefix(cluster_id):
  if cluster_id == 'mpi':
    return '/lustre/home/jsingh/projects'
  elif cluster_id == 'alpha' or cluster_id == 'capella':
    return '/projects/p_neurasearch'
  else:
    raise ValueError('Unsupported argument `cluster_id`.')


def get_jobfile_path(cfg, eval, dd):
  lr_ext = str(cfg.lr).replace(".", "p")
  return f'{get_cluster_prefix(cfg.cluster_id)}/fastlm/execs/{cfg.arch_id}/{cfg.n}/eval-job_gbs-{cfg.gbs}_lr-{lr_ext}_d-{dd}__{eval}.sub'


def get_config_path(cfg, dd):
  lr_ext = str(cfg.lr).replace(".", "p")
  return f'{get_cluster_prefix(cfg.cluster_id)}/fastlm/execs/{cfg.arch_id}/{cfg.n}/eval-job_gbs-{cfg.gbs}_lr-{lr_ext}_d-{dd}.yaml'


def get_jobfile_content(cfg, n_jobs, eval, dd):
  if cfg.cluster_id == 'mpi':
    return f"""# Executable should be a full path
executable=/home/jsingh/projects/fastlm/cluster/single_gpu/eval_condor.sh

# Hyperparmeters are specified in a YAML configuration file
config={get_config_path(cfg, dd)}

# Queue as many jobs as points in the hyperaparameter grid
n_jobs={n_jobs}
eval={eval}

# Pass arguments to the executable
arguments = $(eval) $(config) $(Process) $(Cluster)

# Logs
LOGS_DIR=/fast/jsingh/logs/fastlm/june/attn

error = $(LOGS_DIR)/err/job.$(Cluster).$(Process).err
output = $(LOGS_DIR)/out/job.$(Cluster).$(Process).out
log = $(LOGS_DIR)/log/job.$(Cluster).$(Process).log

# Job requirements
request_memory = 32G
request_cpus = 8
request_gpus = 1
requirements = (TARGET.CUDADeviceName == "NVIDIA A100-SXM4-80GB" || TARGET.CUDADeviceName == "NVIDIA H100 80GB HBM3" || TARGET.CUDADeviceName == "NVIDIA H100") && (Machine != "g174.internal.cluster.is.localnet")

queue $(n_jobs)
  """


def main(cli_cfg: Config):
  print("Started eval management")
  dd = cli_cfg.d
  cfg = parse_cfg(cli_cfg)
  subb = cfg.submit
  n_jobs = 1

  os.makedirs(f'{get_cluster_prefix(cfg.cluster_id)}/fastlm/execs/{cfg.arch_id}/{cfg.n}', exist_ok=True)


  ds = [x.strip() for x in cfg.d.split(',')]
  if ',' in cfg.d:
    n_jobs = len(ds)

  config_dict = vars(cfg)
  config_dict.pop('submit')
  config_dict['d'] = ds

  config_path = get_config_path(cfg, dd)
  with open(config_path, 'w') as f:
    yaml.safe_dump(config_dict, f)

  evals = ['dclm_core', 'ruler']
  for eval in evals:
    jobfile_content = get_jobfile_content(cfg, n_jobs, eval, dd)
    jobfile_path = get_jobfile_path(cfg, eval, dd)

    with open(jobfile_path, 'w') as f:
      f.write(jobfile_content)

    if subb == 'yes':
      # set up the batch-job command according to the cluster we're on
      if cfg.cluster_id == 'mpi':
        cmdlist = ['condor_submit_bid', f'{cfg.bid}', jobfile_path]
      elif cfg.cluster_id in ['capella', 'alpha']:
        cmdlist = ['sbatch', jobfile_path]
      else:
        raise ValueError('Unsupported value found for `--cluster_id`.')

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

        print(eval)
        print(config_path)
        print(jobfile_path)
        print(cluster_sub_id)

      else:
        print('Something bad happened when we submit the job using subprocess! Printing the subprocess call error:')
        print(result.stderr)


if __name__ == '__main__':
  cfg = tyro.cli(Config)
  main(cfg)

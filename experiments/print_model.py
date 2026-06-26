import time
from collections import defaultdict
from contextlib import suppress

import torch
from absl import app, flags
from torch.utils.flop_counter import FlopCounterMode

from src import utils
from src.checkpoint_utils import create_save_steps, load_metrics_from_checkpoint, maybe_load_checkpoint, save_checkpoint
from src.data import get_dataloaders
from src.engine import TorchEngine
from src.models import construct_model
from src.torch_utils import destroy_ddp, pytorch_setup
from src.utils import print_master

flags.DEFINE_string('config', 'src/config/cfg_test.yaml', 'Path to config.yaml file.')
flags.DEFINE_integer('job_idx', None, 'Job idx for job-array sweeps. From 0 to n-1.')
flags.DEFINE_integer('job_cluster', None, 'Job cluster ID.')
FLAGS = flags.FLAGS


def main(argv):
  CFG_PATH = FLAGS.config
  cfg, _ = utils.load_config(CFG_PATH)

  local_rank, world_size, device, master_process = pytorch_setup(cfg)
  utils.set_arch(cfg)
  utils.set_batch_sizes(cfg, world_size)
  utils.set_token_budget_id_from_gbs(cfg)

  print_master(f'Training an {cfg.arch_id.upper()} [{cfg.param_scale_id}] LLM on {cfg.token_budget_id} tokens.')
  print_master(
    f'Training on {world_size} GPUs, GBS={cfg.global_batch_size}, MBS={cfg.micro_batch_size}, GAS={cfg.grad_accumulation_steps}.'
  )

  if cfg.use_wandb and master_process:
    utils.init_wandb(cfg)
    utils.log_job_info()

  # Load checkpoint
  ckpt = maybe_load_checkpoint(cfg, world_size)
  if ckpt is not None:
    cfg.resume_step = ckpt['step']

  # Dataset
  trainloader, validloader = get_dataloaders(cfg)

  # Model
  model, _ = construct_model(cfg)
  non_embed_params = model.count_params(non_embedding=True)
  total_params = model.count_params(non_embedding=False)
  print_master(
    f'Initialized model with {round(non_embed_params / 1e6, 2)}M non-embedding params, or, {round(total_params / 1e6)}M total params'
  )
  print(model)


if __name__ == '__main__':
  app.run(main)

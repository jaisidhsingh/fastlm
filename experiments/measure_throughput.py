import random
import time
from collections import defaultdict
from contextlib import nullcontext, suppress

import numpy as np
import torch
from absl import app, flags
from torch.utils.flop_counter import FlopCounterMode

from src import utils
from src.data import get_dataloaders
from src.engine import TorchEngine
from src.models import construct_model
from src.utils.base_utils import print_master
from src.utils.throughput_utils import ThroughputMeasurement, parse_throughput_metrics
from src.utils.torch_utils import destroy_ddp, pytorch_setup

flags.DEFINE_string('config', 'src/config/cfg_test.yaml', 'Path to config.yaml file.')
flags.DEFINE_string('use_flex', 'no', 'Use flex attention.')
flags.DEFINE_string('use_intra_doc_masking', 'no', 'Use packing-based intra-doc masking.')
flags.DEFINE_integer('job_idx', None, 'Job idx for job-array sweeps. From 0 to n-1.')
flags.DEFINE_integer('job_cluster', None, 'Job cluster ID.')
FLAGS = flags.FLAGS


def setup(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.allow_tf32 = True


def main(argv):
  CFG_PATH = FLAGS.config
  cfg, _ = utils.load_config(CFG_PATH)
  setup(cfg.seed)

  cfg.measure_throughput = True

  if FLAGS.use_flex == 'no':
    cfg.use_flex_attention = False
  else:
    cfg.use_flex_attention = True

  if FLAGS.use_intra_doc_masking == 'no':
    cfg.intra_doc_masking = False
  else:
    cfg.intra_doc_masking = True

  print_master('Measuring throughput under the settings:')
  print_master(f'Use flex/optimal attention: {cfg.use_flex_attention}')
  print_master(f'Use intra-doc masking: {cfg.intra_doc_masking}')
  print_master(' ')

  local_rank, world_size, device, master_process = pytorch_setup(cfg)
  utils.set_arch(cfg)
  utils.set_batch_sizes(cfg, world_size)
  utils.set_token_budget_id_from_gbs(cfg)
  ckpt = None

  print_master(f'Measuring throughput of {cfg.arch_id.upper()} [{cfg.param_scale_id}] LLM')
  print_master(
    f'Training on {world_size} GPUs, GBS={cfg.global_batch_size}, MBS={cfg.micro_batch_size}, GAS={cfg.grad_accumulation_steps}.'
  )

  if cfg.use_wandb and master_process:
    utils.init_wandb(cfg)
    utils.log_job_info()

  trainloader, validloader = get_dataloaders(cfg)

  model, _ = construct_model(cfg)
  non_embed_params = model.count_params(non_embedding=True)
  total_params = model.count_params(non_embedding=False)
  print_master(
    f'Initialized model with {round(non_embed_params / 1e6, 2)}M non-embedding params, or, {round(total_params / 1e6)}M total params'
  )

  steps_budget = utils.get_steps_budget(cfg, world_size)
  cfg.steps_budget = steps_budget
  engine = TorchEngine(model, cfg, device, local_rank, ckpt)

  micro_step_budget = steps_budget * cfg.grad_accumulation_steps
  utils.set_eval_every(cfg)

  if micro_step_budget > len(trainloader):
    raise ValueError('trainloader too short!')

  step_start = 0
  micro_step_start = step_start * cfg.grad_accumulation_steps

  metrics = defaultdict(list)
  if ckpt is not None:
    metrics = defaultdict(list, load_metrics_from_checkpoint(cfg, world_size, cluster_id))

  train_loss_array = []
  throughput_ctx = (
    ThroughputMeasurement(cfg, engine.model, non_embed_params) if cfg.measure_throughput else nullcontext()
  )
  stop = False

  print('Starting training.')
  for micro_step, micro_batch in enumerate(trainloader, micro_step_start + 1):
    step = micro_step // cfg.grad_accumulation_steps
    is_step = micro_step % cfg.grad_accumulation_steps == 0

    if (step > steps_budget and is_step) or stop:
      break

    with throughput_ctx:
      train_loss = engine.step(micro_batch)

    train_loss_array.append(train_loss)
    valid_loss = None

    if master_process and step % cfg.log_every_steps == 0 and is_step:
      throughput_metrics = throughput_ctx.get_metrics()
      if throughput_metrics is not None:
        stop = True

      utils.log(
        cfg,
        metrics,
        micro_step,
        train_loss,
        train_loss_array,
        valid_loss,
        engine.optimizer,
        world_size,
        throughput_metrics,
      )
      train_loss_array = []

      if stop:
        print_master(' ')
        for k, v in parse_throughput_metrics(throughput_metrics, cfg, world_size).items():
          print_master(f'{k} \t {v}')

  print_master('=== Throughput analysis steps completed! ===')
  print(' ')
  destroy_ddp()


if __name__ == '__main__':
  app.run(main)

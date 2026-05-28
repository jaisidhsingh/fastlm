import json
import time
import typing as tp
from collections import namedtuple
from contextlib import suppress
from copy import deepcopy
from dataclasses import asdict

import torch
from absl import app, flags
from torch.utils.flop_counter import FlopCounterMode

from src import utils
from src.data import get_dataloaders
from src.engine import TorchEngine
from src.models import construct_model
from src.torch_utils import destroy_ddp, pytorch_setup
from src.utils import GPU_PEAK_FLOPS_PER_SEC_MAP, print_master

flags.DEFINE_string('config', 'config/config.yaml', 'Path to config.yaml file.')
flags.DEFINE_integer('steps', 50, 'How many steps to check OOM for.')
flags.DEFINE_integer('job_idx', None, 'Job idx for job-array sweeps. From 0 to n-1.')
flags.DEFINE_integer('job_cluster', None, 'Job cluster ID.')
FLAGS = flags.FLAGS

BATCH_SIZES = [0, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 72, 84, 96, 128]


def find_max_mbs(cfg: tp.Any, engine: TorchEngine, steps: int = 50, world_size: int = 1):
  mbs_list = deepcopy(BATCH_SIZES)
  mbs_list.reverse()
  max_mbs = mbs_list[0]
  mbs_wise_throughput_metrics = {}

  for bs in mbs_list[1:]:
    config_dict = cfg._asdict()
    config_dict['micro_batch_size'] = bs
    cfg = namedtuple('Config', config_dict.keys())(**config_dict)
    assert cfg.micro_batch_size == bs

    trainloader, validloader = get_dataloaders(cfg)

    try:
      flops_per_micro_step = 0
      step_time = 0
      throughput_metrics = {}

      for micro_step, micro_batch in enumerate(trainloader, 1):
        step = micro_step // cfg.grad_accumulation_steps
        is_step = micro_step % cfg.grad_accumulation_steps == 0
        if step > steps and is_step:
          break

        if micro_step == 1:
          flop_counter = FlopCounterMode(engine.model, display=False, depth=2)
        else:
          flop_counter = suppress()

        torch.cuda.synchronize()
        start = time.perf_counter()

        with flop_counter:
          train_loss = engine.step(micro_batch)
        if micro_step == 1:
          flops_per_micro_step = flop_counter.get_total_flops()

        torch.cuda.synchronize()
        end = time.perf_counter()
        micro_step_time = end - start
        step_time += micro_step_time

        if cfg.eval and step % cfg.eval_every_steps == 0 and is_step:
          print_master('Evaluating on validation set')
          valid_loss = engine.eval(validloader)

        if is_step:
          tokens_per_step = cfg.batch_size * cfg.seq_len * cfg.grad_accumulation_steps * world_size
          tokens_per_sec = tokens_per_step / step_time
          flops_per_step = flops_per_micro_step * cfg.grad_accumulation_steps
          flops_per_sec = flops_per_step / step_time
          mfu = flops_per_sec / GPU_PEAK_FLOPS_PER_SEC_MAP[torch.cuda.get_device_name(0) * world_size]
          throughput_metrics[step] = {
            'tokens_per_sec': tokens_per_sec,
            'flops_per_sec': flops_per_sec,
            'step_time': step_time,
            'mfu': mfu,
          }
          step_time = 0

      max_mbs = bs
      mbs_wise_throughput_metrics[str(max_mbs)] = throughput_metrics
      return max_mbs, mbs_wise_throughput_metrics

    except torch.cuda.OutOfMemoryError:
      continue

  return max_mbs, mbs_wise_throughput_metrics


def main(_):
  CFG_PATH = FLAGS.config
  cfg, _ = utils.load_config(CFG_PATH)

  print_master(f'Starting pretraining experiment: {cfg.exp_name}')

  local_rank, world_size, device, master_process = pytorch_setup(cfg)
  print_master(f'Number of GPUs: {world_size}')

  model, _ = construct_model(cfg)

  engine = TorchEngine(model, cfg, device, local_rank, ckpt=None)
  max_mbs, mbs_wise_throughput_metrics = find_max_mbs(cfg, engine, FLAGS.steps, world_size=world_size)

  result = {}
  result['max_mbs'] = max_mbs
  result['mbs_wise_throughput_metrics'] = mbs_wise_throughput_metrics
  result['config'] = cfg._asdict()
  result['gpu_name'] = torch.cuda.get_device_name(0)
  result['world_size'] = world_size
  print(result)
  destroy_ddp()


if __name__ == '__main__':
  app.run(main)

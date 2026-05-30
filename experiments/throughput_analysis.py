import json
import time
import typing as tp
from collections import namedtuple
from contextlib import suppress
from copy import deepcopy
from dataclasses import asdict
from statistics import mean, stdev
from types import SimpleNamespace

import torch
from absl import app, flags
from torch.cuda import OutOfMemoryError
from torch.utils.flop_counter import FlopCounterMode

from src import utils
from src.data import get_dataloaders
from src.engine import TorchEngine
from src.models import construct_model
from src.torch_utils import destroy_ddp, pytorch_setup
from src.utils import GPU_PEAK_FLOPS_PER_SEC_MAP, print_master

flags.DEFINE_string('param_scale_id', '300M', 'ID of the parameter scale for our models.')
flags.DEFINE_integer('micro_batch_size', 32, 'Micro batch size.')
flags.DEFINE_integer('grad_accumulation_steps', 4, 'Gradient accumulation steps.')
flags.DEFINE_integer('steps', 100, 'How many steps to check OOM for.')
FLAGS = flags.FLAGS


def main(_):
  cfg = utils.load_config_from_constants(flags.param_scale_id)
  cfg.micro_batch_size = flags.micro_batch_size
  cfg.grad_accumulation_steps = flags.grad_accumulation_steps
  cfg.steps_budget = 1000

  local_rank, world_size, device, master_process = pytorch_setup(cfg)
  print_master(f'Number of GPUs: {world_size}')

  model, model_cfg = construct_model(cfg)
  done = False

  while not done:
    flops_per_micro_step = 0
    step_time = 0
    throughput_metrics = {}

    try:
      engine = TorchEngine(model, cfg, device, local_rank, ckpt=None)
      trainloader, validloader = get_dataloaders(cfg)

      for micro_step, micro_batch in enumerate(trainloader, 1):
        step = micro_step // cfg.grad_accumulation_steps
        is_step = micro_step % cfg.grad_accumulation_steps == 0
        if step > flags.steps and is_step:
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

        if step == flags.steps and is_step:
          print_master('Evaluating on validation set')
          valid_loss = engine.eval(validloader)

        if is_step:
          tokens_per_step = cfg.micro_batch_size * cfg.seq_len * cfg.grad_accumulation_steps * world_size
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

      est = {
        'param_scale_id': flags.param_scale_id,
        'steps_run': flags.steps,
        'final_valid_loss': valid_loss,
        'micro_batch_size': cfg.micro_batch_size,
        'grad_accumulation_steps': cfg.grad_accumulation_steps,
      }
      for k in throughput_metrics[step].keys():
        mlist = [throughput_metrics[flags.steps - i][k] for i in range(flags.steps // 2)]
        est[k] = {'mean': mean(mlist), 'std': stdev(mlist)}

      print_master(est)
      done = True

    except torch.cuda.OutOfMemoryError:
      cfg.micro_batch_size = cfg.micro_batch_size // 2
      cfg.grad_accumulation_steps = int(cfg.grad_accumulation_steps * 2)

      if cfg.micro_batch_size == 1:
        done = True
      continue

  destroy_ddp()


if __name__ == '__main__':
  app.run(main)

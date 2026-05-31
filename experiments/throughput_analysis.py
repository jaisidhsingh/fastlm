import json
import os
import time
from contextlib import suppress
from statistics import mean, stdev

import torch
from absl import app, flags
from torch.utils.flop_counter import FlopCounterMode
from tqdm import tqdm

from src import utils
from src.constants import SCALING_RESULTS_FOLDER
from src.data import get_dataloaders
from src.engine import TorchEngine
from src.models import construct_model
from src.torch_utils import destroy_ddp, pytorch_setup
from src.utils import GPU_PEAK_FLOPS_PER_SEC_MAP, print_master

flags.DEFINE_string('arch_id', 'attn', 'Architecture of the LLM.')
flags.DEFINE_string('param_scale_id', '300M', 'ID of the parameter scale for our models.')
flags.DEFINE_integer('global_batch_size', 32, 'Micro batch size.')
flags.DEFINE_integer('steps', 10, 'How many steps to check OOM for.')
args = flags.FLAGS


def main(argv):
  cfg = utils.load_config_from_constants(args.param_scale_id)
  local_rank, world_size, device, master_process = pytorch_setup(cfg)

  save_folder = os.path.join(
    SCALING_RESULTS_FOLDER, args.arch_id, args.param_scale_id, 'gbs_wise_results', f'gbs_{args.global_batch_size}'
  )
  os.makedirs(save_folder, exist_ok=True)

  arch, ratio = utils.parse_arch_id(args.arch_id)
  cfg.token_mixer = arch
  cfg.hybrid_mixer_ratio = ratio

  print_master(f'Architecture={args.arch_id}:{args.param_scale_id} and GBS={args.global_batch_size}')

  cfg.micro_batch_size = args.global_batch_size // world_size
  cfg.grad_accumulation_steps = 1
  cfg.steps_budget = 1000

  print_master(f'Number of GPUs: {world_size}')
  print_master('Starting throughput analysis\n')

  model, model_cfg = construct_model(cfg)
  total_params = model.count_params(non_embedding=False)
  non_embedding_params = model.count_params(non_embedding=True)
  print_master(
    f'Initilized model with total params={round(total_params / 1e6, 2)}M, non-embedding params={round(non_embedding_params / 1e6, 2)}M'
  )
  done = False
  est = None

  while not done:
    flops_per_micro_step = 0
    step_time = 0
    throughput_metrics = {}

    print_master(
      f'For GBS={args.global_batch_size}, checking MBS={cfg.micro_batch_size}, GAS={cfg.grad_accumulation_steps}'
    )

    try:
      engine = TorchEngine(model, cfg, device, local_rank, ckpt=None)
      trainloader, validloader = get_dataloaders(cfg)

      for micro_step, micro_batch in enumerate(trainloader, 1):
        step = micro_step // cfg.grad_accumulation_steps
        is_step = micro_step % cfg.grad_accumulation_steps == 0
        if step > args.steps and is_step:
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

        if step == args.steps and is_step:
          print_master('Evaluating on validation set')
          valid_loss = engine.eval(validloader)

        if is_step:
          tokens_per_step = cfg.micro_batch_size * cfg.seq_len * cfg.grad_accumulation_steps * world_size
          tokens_per_sec = tokens_per_step / step_time
          flops_per_step = flops_per_micro_step * cfg.grad_accumulation_steps * world_size
          flops_per_sec = flops_per_step / step_time
          mfu = flops_per_sec / GPU_PEAK_FLOPS_PER_SEC_MAP[torch.cuda.get_device_name(0) * world_size]

          throughput_metrics[step] = {
            'flops_per_step': flops_per_step,
            'tokens_per_sec': tokens_per_sec,
            'flops_per_sec': flops_per_sec,
            'step_time': step_time,
            'mfu': mfu,
          }
          step_time = 0
          print(f'Step={step}, tokens_per_sec={tokens_per_sec}')

      est = {
        'param_scale_id': args.param_scale_id,
        'steps_run': args.steps,
        'final_valid_loss': valid_loss,
        'global_batch_size': int(cfg.micro_batch_size * cfg.grad_accumulation_steps * world_size),
        'micro_batch_size': cfg.micro_batch_size,
        'grad_accumulation_steps': cfg.grad_accumulation_steps,
        'world_size_for_throughput_study': world_size,
        'throughput_metrics': {},
      }
      for k in throughput_metrics[step - 1].keys():
        mlist = [throughput_metrics[args.steps - i][k] for i in range(args.steps // 2)]
        est['throughput_metrics'][k] = {'mean': mean(mlist), 'std': stdev(mlist)}

      print_master(
        f'For GBS={args.global_batch_size}, MBS={cfg.micro_batch_size}, GAS={cfg.grad_accumulation_steps} works!'
      )
      print_master(f'Estimated tokens-per-second={est["throughput_metrics"]["tokens_per_sec"]["mean"]}')
      print_master('Saving throughput logs and exiting.')
      with open(os.path.join(save_folder, 'throughput_analysis.json'), 'w') as f:
        json.dump(est, f)
      done = True

    except torch.cuda.OutOfMemoryError:
      print_master(f'MBS={cfg.micro_batch_size} does not fit in memory. Reducing by half and doubling GAS...\n')
      cfg.micro_batch_size = cfg.micro_batch_size // 2
      cfg.grad_accumulation_steps = int(cfg.grad_accumulation_steps * 2)

      if cfg.micro_batch_size == 0:
        print_master('Cannot fit any samples in memory. Nothing to save. Exiting.')
        done = True

  if est is not None:
    print('Results saved at', os.path.join(save_folder, 'throughput_analysis.json'))
  destroy_ddp()


if __name__ == '__main__':
  # app.run(main)
  # folder = '/fast/jsingh/projects/fastlm/june/results/attn'

  p = lambda i, b: (
    f'/fast/jsingh/projects/fastlm/june/results/attn/{i}/gbs_wise_results/gbs_{b}/throughput_analysis.json'
  )

  for ids in ['20M', '50M', '150M', '300M']:
    for gbs in [16, 32, 64, 128, 256, 512]:
      if not os.path.exists(p(ids, gbs)):
        print(ids, gbs)

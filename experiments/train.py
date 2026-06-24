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

  # Engine
  steps_budget = utils.get_steps_budget(cfg, world_size)
  cfg.steps_budget = steps_budget
  engine = TorchEngine(model, cfg, device, local_rank, ckpt)

  if cfg.scheduler == 'linear_cooldown':
    steps_budget = cfg.resume_step + engine.scheduler.cooldown_steps
    cfg.steps_budget = steps_budget

  print_master(
    f'Training will run for {steps_budget} steps, or, {steps_budget * cfg.grad_accumulation_steps} micro steps'
  )

  # How long do we wanna train for (takes into account resume + cooldown)
  micro_step_budget = steps_budget * cfg.grad_accumulation_steps
  utils.set_eval_every(cfg)

  if micro_step_budget > len(trainloader):
    raise ValueError('trainloader too short!')
  print_master(f'Training for {steps_budget} steps <=> {micro_step_budget} micro_steps')
  print_master(f'Evaluating every {cfg.eval_every_steps} for a total of {cfg.num_evals} eval points.')

  # Start the dataloader, but remember that the 2nd argument to enumerate only changes the index value.
  # It DOES NOT skip over that many batches in the dataloader.
  step_start = 0
  micro_step_start = step_start * cfg.grad_accumulation_steps
  print_master(
    f'=== Start Training from step: {step_start}/{steps_budget}, micro_step: {micro_step_start}/{micro_step_budget} ==='
  )

  # Bookkeeping
  metrics = defaultdict(list)
  if ckpt is not None:
    metrics = defaultdict(list, load_metrics_from_checkpoint(cfg, world_size))

  train_loss_array = []

  # Bookkeeping for throughput
  flops_per_micro_step = 0
  step_time = 0

  resume_step = None
  if ckpt is not None:
    resume_step = ckpt['step']
    cfg.save_last_checkpoint = False
    reached_new_data = False
    print_master(f'Resuming state from step={resume_step}, cooldown only={cfg.cooldown_only}')

  # When do we want to save
  # exp_folder = utils.get_exp_dir_path(cfg, world_size)
  save_points = create_save_steps(cfg, world_size)
  assert save_points is not None, "Save tracking is incorrect, & this is a problem even if we're not saving anything."

  # Training
  for micro_step, micro_batch in enumerate(trainloader, micro_step_start + 1):
    step = micro_step // cfg.grad_accumulation_steps
    is_step = micro_step % cfg.grad_accumulation_steps == 0

    if resume_step is not None:
      if step <= resume_step:
        continue
      else:
        if not reached_new_data:
          steps_budget += resume_step
          reached_new_data = True

    if step > steps_budget and is_step:
      break

    # count FLOPs used
    if micro_step == 1 and cfg.measure_throughput:
      flop_counter = FlopCounterMode(model, display=False, depth=2)
    else:
      flop_counter = suppress()

    if cfg.measure_throughput:
      torch.cuda.synchronize()
      start = time.perf_counter()

    # Train
    with flop_counter:
      train_loss = engine.step(micro_batch)
    if micro_step == 1 and cfg.measure_throughput:
      flops_per_micro_step = flop_counter.get_total_flops()
    train_loss_array.append(train_loss)

    if cfg.measure_throughput:
      torch.cuda.synchronize()
      end = time.perf_counter()
      micro_step_time = end - start
      step_time += micro_step_time

    # Eval
    valid_loss = None
    if cfg.eval and is_step:
      if step % cfg.eval_every_steps == 0 or step == steps_budget:  # last step
        print_master('Evaluating on validation set')
        valid_loss = engine.eval(validloader)

    # Log
    if master_process and step % cfg.log_every_steps == 0 and is_step:
      throughput_metrics = None
      if cfg.measure_throughput:
        flops_per_step = flops_per_micro_step * cfg.grad_accumulation_steps
        throughput_metrics = (step_time, flops_per_step)

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
      if cfg.measure_throughput:
        step_time = 0

    # Checkpoint
    if master_process and is_step and step in save_points and cfg.save_intermediate_checkpoints:
      save_name = save_points[step]
      save_checkpoint(step, model, engine, cfg, metrics, save_name, world_size)

  # End of training: log and save checkpoint
  print_master('=== Training Completed! ===')
  if master_process and cfg.save_last_checkpoint:
    save_checkpoint(step, model, engine, cfg, metrics, 'end_of_training_backup', world_size)

  # DDP slaughtering
  destroy_ddp()


if __name__ == '__main__':
  app.run(main)

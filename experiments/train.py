import time
from collections import defaultdict
from contextlib import nullcontext, suppress

import torch
from absl import app, flags
from torch.utils.flop_counter import FlopCounterMode

from src import utils
from src.engine import TorchEngine
from src.models import construct_model
from src.utils.base_utils import print_master
from src.utils.checkpoint_utils import (
  create_save_steps,
  load_metrics_from_checkpoint,
  maybe_load_checkpoint,
  save_checkpoint,
)
from src.utils.throughput_utils import ThroughputMeasurement
from src.utils.torch_utils import destroy_ddp, pytorch_setup

flags.DEFINE_string('config', 'src/config/cfg_test.yaml', 'Path to config.yaml file.')
flags.DEFINE_integer('job_idx', None, 'Job idx for job-array sweeps. From 0 to n-1.')
flags.DEFINE_integer('job_cluster', None, 'Job cluster ID.')
flags.DEFINE_string('cluster_id', None, 'Which cluster are we running things on?')
FLAGS = flags.FLAGS


def main(argv):
  CFG_PATH = FLAGS.config
  CLUSTER_ID = FLAGS.cluster_id
  cfg, _ = utils.load_config(CFG_PATH)

  local_rank, world_size, device, master_process = pytorch_setup(cfg)
  utils.set_arch(cfg)
  utils.set_batch_sizes(cfg, world_size, CLUSTER_ID)
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

  print_master(
    f'Training till {steps_budget} step point, or, {steps_budget * cfg.grad_accumulation_steps} micro step point'
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

  # Setup for resuming
  resume_step = None
  if ckpt is not None:
    resume_step = ckpt['step']
    cfg.save_last_checkpoint = False
    reached_new_data = False
    print_master(f'Resuming state from step={resume_step}, cooldown only={cfg.cooldown_only}')

  # When do we want to save
  save_points, save_toks = create_save_steps(cfg, world_size)
  assert save_points is not None and save_toks is not None, (
    "Save tracking is incorrect, & this is a problem even if we're not saving anything."
  )
  print('Saving at step points...')
  for k, v in save_points.items():
    print(f'Step point: {k}, save name: {v}')
  print('Saving at token points', save_toks)

  # Bookkeeping
  metrics = defaultdict(list)
  if ckpt is not None:
    metrics = defaultdict(list, load_metrics_from_checkpoint(cfg, world_size))

  train_loss_array = []
  throughput_ctx = ThroughputMeasurement(cfg, engine.model) if cfg.measure_throughput else nullcontext()

  # Training
  for micro_step, micro_batch in enumerate(trainloader, micro_step_start + 1):
    step = micro_step // cfg.grad_accumulation_steps
    is_step = micro_step % cfg.grad_accumulation_steps == 0

    # Sampler is sequential, so find new data.
    if resume_step is not None:
      if step <= resume_step:
        continue
      else:
        if not reached_new_data:
          reached_new_data = True

    # Stop-training boundary
    if step > steps_budget and is_step:
      break

    # Train a step
    with throughput_ctx:
      train_loss = engine.step(micro_batch)
    train_loss_array.append(train_loss)

    # Validation loop
    valid_loss = None
    if cfg.eval and is_step:
      if step % cfg.eval_every_steps == 0 or step == steps_budget:  # last step
        print_master('Evaluating on validation set')
        valid_loss = engine.eval(validloader)

    # Log
    if master_process and step % cfg.log_every_steps == 0 and is_step:
      throughput_metrics = throughput_ctx.get_metrics()

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

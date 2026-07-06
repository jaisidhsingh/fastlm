"""
End-to-end training profiler.

Runs a short training loop mimicking `experiments/train.py` and profiles every phase:
  1. Data loading (DataLoader iterator overhead)
  2. Host→device transfer (_move_to_device)
  3. Forward pass (model + loss)
  4. Backward pass
  5. Optimizer step (grad unscaling, clipping, stepping, zeroing)

Outputs:
  - A sorted table of top GPU kernel time by category/name.
  - A Chrome trace file (`profile_trace.json`) for `chrome://tracing`.
  - Actionable bottleneck recommendations.

Usage:
  python testing/profile_test.py --config src/config/cfg_test.yaml
"""

import os
import sys
import time
from collections import defaultdict

import torch
from absl import app, flags
from torch.profiler import ProfilerActivity, profile, record_function

# ---------------------------------------------------------------------------
# Path wrangling so we can import from `src` and run this file directly
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
  sys.path.insert(0, _PROJECT_ROOT)

from src import utils
from src.data import get_dataloaders
from src.engine import TorchEngine
from src.models import construct_model
from src.torch_utils import destroy_ddp, pytorch_setup
from src.utils import print_master

# ---------------------------------------------------------------------------
# Flags (matching experiments/train.py so that utils.load_config works)
# ---------------------------------------------------------------------------
flags.DEFINE_string(
  'config', os.path.join(_PROJECT_ROOT, 'src', 'config', 'cfg_test.yaml'), 'Path to config.yaml file.'
)
flags.DEFINE_integer('job_idx', None, 'Job idx for job-array sweeps. From 0 to n-1.')
flags.DEFINE_integer('job_cluster', None, 'Job cluster ID.')
FLAGS = flags.FLAGS

# ---------------------------------------------------------------------------
# Hard-coded profiling parameters
# ---------------------------------------------------------------------------
NUM_WARMUP_STEPS = 2000  # warmup for torch.compile + CUDA allocator settling
NUM_PROFILE_STEPS = 100  # steps to profile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _gpu_name() -> str:
  if torch.cuda.is_available():
    return torch.cuda.get_device_name(0)
  return 'unknown'


def _format_time(seconds: float) -> str:
  if seconds >= 1.0:
    return f'{seconds:.3f} s'
  elif seconds >= 1e-3:
    return f'{seconds * 1e3:.2f} ms'
  else:
    return f'{seconds * 1e6:.1f} µs'


# ---------------------------------------------------------------------------
# Phase 1 – Data-loading micro-benchmark
# ---------------------------------------------------------------------------
def profile_dataloader(trainloader, device, num_batches: int = 50):
  """Time how long it takes to fetch batches (excludes GPU work)."""
  print('\n' + '=' * 70)
  print('PHASE 1 — DataLoader throughput')
  print('=' * 70)

  fetch_times = []
  for i, batch in enumerate(trainloader):
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # Simulate moving inputs to device (excludes forward/backward)
    if isinstance(batch, dict):
      inp = batch['input_ids']
    else:
      inp = batch
    if hasattr(inp, 'pin_memory'):
      inp = inp.pin_memory().to(device, non_blocking=True)
    torch.cuda.synchronize()

    t1 = time.perf_counter()
    fetch_times.append(t1 - t0)

    if i + 1 >= num_batches:
      break

  fetch_times_sorted = sorted(fetch_times)
  avg = sum(fetch_times) / len(fetch_times)
  p50 = fetch_times_sorted[len(fetch_times_sorted) // 2]
  p99 = fetch_times_sorted[int(len(fetch_times_sorted) * 0.99)]

  print(f'  Batches sampled:  {len(fetch_times)}')
  print(f'  Avg fetch+transfer: {_format_time(avg)}')
  print(f'  P50 fetch+transfer: {_format_time(p50)}')
  print(f'  P99 fetch+transfer: {_format_time(p99)}')
  return avg


# ---------------------------------------------------------------------------
# Phase 2 – Full training-step profiling with torch.profiler
# ---------------------------------------------------------------------------
def profile_training(trainloader, engine, cfg, device, num_warmup: int, num_profile: int):
  print('\n' + '=' * 70)
  print('PHASE 2 — Training-step profile (torch.profiler)')
  print('=' * 70)
  print(f'  Warmup steps:  {num_warmup}')
  print(f'  Profile steps: {num_profile}')
  print(f'  GPU:           {_gpu_name()}')
  print(f'  torch.compile: {cfg.torch_compile}')

  # ---- warmup -----------------------------------------------------------
  train_iter = iter(trainloader)
  for i in range(num_warmup):
    batch = next(train_iter)
    engine.step(batch)
  torch.cuda.synchronize()

  # ---- profiled loop ----------------------------------------------------
  activities = [ProfilerActivity.CPU]
  if 'cuda' in device:
    activities.append(ProfilerActivity.CUDA)

  prof = profile(
    activities=activities,
    record_shapes=True,
    with_stack=True,
    profile_memory=False,
    with_modules=True,
  )

  prof.start()
  for i in range(num_profile):
    with record_function(f'training_step_{i}'):
      batch = next(train_iter)
      with record_function('engine.step'):
        engine.step(batch)
    torch.cuda.synchronize()
  prof.stop()

  # ---- output -----------------------------------------------------------
  trace_path = os.path.join(_THIS_DIR, 'profile_trace.json')
  prof.export_chrome_trace(trace_path)
  print(f'\n  Chrome trace saved → {trace_path}')

  # ---- table ------------------------------------------------------------
  print('\n' + '-' * 70)
  print('Top GPU kernels by total CUDA time (self):')
  print('-' * 70)
  try:
    key_events = prof.key_averages(group_by_input_shape=True)
  except TypeError:
    key_events = prof.key_averages()

  table = key_events.table(
    sort_by='cuda_time_total',
    row_limit=30,
    header='Name | CPU total | CPU self | CUDA total | CUDA self | # calls',
  )
  print(table)

  # ---- aggregated category breakdown ------------------------------------
  print('\n' + '-' * 70)
  print('Aggregated by top-level operation category:')
  print('-' * 70)
  cat_totals: dict[str, float] = defaultdict(float)
  for evt in key_events:
    # group by the first meaningful part of the name
    name = evt.key
    # Strip parameterised suffixes for cleaner grouping
    if 'aten::' in name:
      cat = 'aten::' + name.split('aten::')[1].split('(')[0].split('_')[0]
    elif 'record_function' in name:
      cat = name
    elif '/' in name:
      cat = name.split('/')[0]
    else:
      cat = name

    cuda_t = getattr(evt, 'cuda_time_total', 0) or 0
    cat_totals[cat] += cuda_t

  sorted_cats = sorted(cat_totals.items(), key=lambda x: x[1], reverse=True)
  total_cuda = sum(v for _, v in sorted_cats)
  for cat, t in sorted_cats[:15]:
    pct = (t / total_cuda * 100) if total_cuda > 0 else 0
    print(f'  {cat:<50s} {_format_time(t):>10s}  ({pct:5.1f}%)')

  return prof


# ---------------------------------------------------------------------------
# Phase 3 – Wall-clock breakdown per phase
# ---------------------------------------------------------------------------
def profile_phases(trainloader, engine, cfg, device, num_steps: int = 10):
  """Break down a training step into data-move / forward / backward / optimiser.

  Uses fresh dataloader iters to avoid interfering with prior phases.
  Collects multiple samples for reliable averages.
  """
  print('\n' + '=' * 70)
  print('PHASE 3 — Wall-clock phase breakdown (instrumented)')
  print('=' * 70)

  from src.engine.engine import _move_to_device

  model = engine.model
  model.train()

  records: list[dict[str, float]] = []

  train_iter = iter(trainloader)
  # Warmup one step to get past any lazy init
  batch = next(train_iter)
  _ = engine.step(batch)
  torch.cuda.synchronize()

  for _ in range(num_steps):
    batch = next(train_iter)

    # 1) Data movement
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    inputs, targets, attn_mask, linear_mask, cu_seqlens = _move_to_device(
      batch, cfg.seq_len, device, cfg.intra_doc_masking
    )
    torch.cuda.synchronize()
    t_data = time.perf_counter() - t0

    # 2) Forward
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with engine.ctx:
      output = model(inputs, attn_mask, linear_mask, cu_seqlens)
      logits = getattr(output, 'logits', output)
      loss = engine.criterion(
        logits[:, :-1, :].reshape(-1, logits.size(-1)).contiguous(),
        targets.reshape(-1).contiguous(),
      )
      loss = loss / engine.accumulation_steps
    torch.cuda.synchronize()
    t_fwd = time.perf_counter() - t0

    # 3) Backward
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    engine.scaler.scale(loss).backward()
    torch.cuda.synchronize()
    t_bwd = time.perf_counter() - t0

    # 4) Optimiser
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if engine.grad_clip:
      engine.scaler.unscale_(engine.optimizer)
      torch.nn.utils.clip_grad_norm_(model.parameters(), engine.grad_clip)
    engine.scaler.step(engine.optimizer)
    engine.scaler.update()
    engine.optimizer.zero_grad(set_to_none=True)
    if engine.scheduler:
      engine.scheduler.step()
    torch.cuda.synchronize()
    t_opt = time.perf_counter() - t0

    records.append({'data': t_data, 'fwd': t_fwd, 'bwd': t_bwd, 'opt': t_opt})

  # ---- average across steps ----------------------------------------------
  avg = {k: sum(r[k] for r in records) / len(records) for k in records[0]}
  total = avg['data'] + avg['fwd'] + avg['bwd'] + avg['opt']

  print(f'  Averages over {num_steps} steps (GPU-synchronised wall-clock):')
  print(f'  {"Phase":<25s} {"Mean":>10s}   {"%":>6s}   {"Min":>10s}   {"Max":>10s}')
  print(f'  {"-" * 70}')
  for name, key in [
    ('1. Data → GPU', 'data'),
    ('2. Forward pass', 'fwd'),
    ('3. Backward pass', 'bwd'),
    ('4. Optimiser step', 'opt'),
  ]:
    vals = [r[key] for r in records]
    pct = (avg[key] / total * 100) if total > 0 else 0
    print(
      f'  {name:<25s} {_format_time(avg[key]):>10s}  {pct:5.1f}%'
      f'   {_format_time(min(vals)):>10s}   {_format_time(max(vals)):>10s}'
    )
  print(f'  {"─" * 70}')
  print(f'  {"Total":<25s} {_format_time(total):>10s}')

  return total, (avg['data'], avg['fwd'], avg['bwd'], avg['opt'])


# ---------------------------------------------------------------------------
# Bottleneck analysis
# ---------------------------------------------------------------------------
def analyse_bottleneck(data_time, fwd_time, bwd_time, opt_time, step_total):
  print('\n' + '=' * 70)
  print('BOTTLENECK ANALYSIS')
  print('=' * 70)

  total = data_time + fwd_time + bwd_time + opt_time
  p_data = data_time / total * 100
  p_fwd = fwd_time / total * 100
  p_bwd = bwd_time / total * 100
  p_opt = opt_time / total * 100

  recommendations = []

  if p_data > 15:
    recommendations.append(
      '⚠️  DATA LOADING is a significant bottleneck ({:.1f}%).\n'
      '    → Increase `num_workers` in your config.\n'
      '    → Use `prefetch_factor` > 2.\n'
      '    → Ensure data is pre-tokenized and stored on fast SSD/ramdisk.\n'
      '    → Consider streaming from memory-mapped files.'.format(p_data)
    )
  else:
    recommendations.append('✅ Data loading is reasonable ({:.1f}% of step time).'.format(p_data))

  if p_fwd > 45:
    recommendations.append(
      '⚠️  FORWARD PASS is the dominant cost ({:.1f}%).\n'
      '    → Ensure `torch.compile` is enabled.\n'
      '    → Consider mixed-precision (bfloat16) if not already used.\n'
      '    → Check attention implementation – FlashAttention is critical.\n'
      '    → Reduce model size or sequence length if acceptable.'.format(p_fwd)
    )
  elif p_fwd > 25:
    recommendations.append('ℹ️  Forward pass is moderate ({:.1f}% of step time).'.format(p_fwd))
  else:
    recommendations.append('✅ Forward pass is efficient ({:.1f}% of step time).'.format(p_fwd))

  if p_bwd > 50:
    recommendations.append(
      '⚠️  BACKWARD PASS is the largest cost ({:.1f}%).\n'
      '    → This is usually normal (bwd ≈ 2× fwd).\n'
      '    → Gradient accumulation can help amortise optimiser cost.\n'
      '    → Ensure activation checkpointing if memory-bound.'.format(p_bwd)
    )
  elif p_bwd > 30:
    recommendations.append('ℹ️  Backward pass is expected ({:.1f}% of step time).'.format(p_bwd))
  else:
    recommendations.append('✅ Backward pass is efficient ({:.1f}% of step time).'.format(p_bwd))

  if p_opt > 20:
    recommendations.append(
      '⚠️  OPTIMISER STEP is expensive ({:.1f}%).\n'
      '    → Use `fused_optim: True` (fused AdamW).\n'
      '    → Increase `grad_accumulation_steps` to amortise.\n'
      '    → Consider 8-bit optimisers if memory-constrained.'.format(p_opt)
    )
  else:
    recommendations.append('✅ Optimiser step is reasonable ({:.1f}% of step time).'.format(p_opt))

  rec = '\n'.join(f'  {r}' for r in recommendations)

  if bwd_time > 0 and fwd_time > 0:
    ratio = bwd_time / fwd_time
    rec += f'\n\n  📊 backward/forward ratio = {ratio:.2f}x'
    if ratio < 1.8:
      rec += '\n     → Lower than typical ~2x. Forward might include compile overheads or data movement.'
    elif ratio > 2.5:
      rec += '\n     → Higher than typical. Possible causes: gradient clipping overhead, large weight grads.'

  print(rec)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv):
  CFG_PATH = FLAGS.config
  cfg, _ = utils.load_config(CFG_PATH)

  local_rank, world_size, device, master_process = pytorch_setup(cfg)
  utils.set_arch(cfg)
  utils.set_batch_sizes(cfg, world_size)
  utils.set_token_budget_id_from_gbs(cfg)

  # Override for profiling: no wandb, no checkpointing, minimal eval
  cfg.use_wandb = False
  cfg.measure_throughput = True
  cfg.print_progress = False
  cfg.eval = False
  cfg.save_last_checkpoint = False
  cfg.save_intermediate_checkpoints = False
  # Keep compile as-is from config (usually True for test)

  print('=' * 70)
  print('TRAINING PROFILER')
  print('=' * 70)
  print(f'  Config:    {CFG_PATH}')
  print(f'  Arch:      {cfg.arch_id}')
  print(f'  Model:     {cfg.param_scale_id}  (d={cfg.d_model}, L={cfg.n_layers}, h={cfg.n_heads})')
  print(f'  Seq len:   {cfg.seq_len}')
  print(f'  MBS:       {cfg.micro_batch_size}')
  print(f'  GAS:       {cfg.grad_accumulation_steps}')
  print(f'  Workers:   {cfg.num_workers}')
  print(f'  Compile:   {cfg.torch_compile}')
  print(f'  dtype:     {cfg.dtype}')

  # ---- dataloaders ----
  print_master('Loading dataset...')
  trainloader, validloader = get_dataloaders(cfg)

  # ---- model ----
  print_master('Building model...')
  model, model_cfg = construct_model(cfg)
  non_embed_params = model.count_params(non_embedding=True)
  total_params = model.count_params(non_embedding=False)
  print_master(f'  Non-embedding params: {non_embed_params / 1e6:.2f}M  |  Total: {total_params / 1e6:.2f}M')

  # ---- engine ----
  steps_budget = utils.get_steps_budget(cfg, world_size)
  cfg.steps_budget = steps_budget
  engine = TorchEngine(model, cfg, device, local_rank, None)

  # ---- 1. DataLoader bench ----
  profile_dataloader(trainloader, device, num_batches=50)

  # ---- 2. Torch profiler ----
  profile_training(trainloader, engine, cfg, device, NUM_WARMUP_STEPS, NUM_PROFILE_STEPS)

  # ---- 3. Phase breakdown ----
  total, (t_data, t_fwd, t_bwd, t_opt) = profile_phases(trainloader, engine, cfg, device, num_steps=NUM_PROFILE_STEPS)

  # ---- 4. Bottleneck analysis ----
  analyse_bottleneck(t_data, t_fwd, t_bwd, t_opt, total)

  # ---- cleanup ----
  destroy_ddp()
  print("\nDone. Open 'profile_trace.json' in chrome://tracing for a visual timeline.")


if __name__ == '__main__':
  app.run(main)

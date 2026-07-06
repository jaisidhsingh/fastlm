import time
from contextlib import suppress

import torch
from torch.utils.flop_counter import FlopCounterMode

GPU_PEAK_FLOPS_PER_SEC_MAP = {
  'NVIDIA A100-SXM4-80GB': 312e12,
  'NVIDIA H100 80GB HBM3': 989e12,
  'NVIDIA H100': 835e12,
}


class ThroughputMeasurement:
  """Context manager for measuring training throughput at the *step* level.
  Skips ``ignore_steps`` training steps (warmup), then measures step time
  and FLOPs across ``measure_steps`` training steps.  Each training step
  consists of ``grad_accumulation_steps`` micro-steps; the context manager
  accumulates the micro-step timings within a step and records the total
  when a step boundary is reached.  Averaged results are available via
  :meth:`get_metrics` once the measurement window is complete.

  Usage::
      throughput = ThroughputMeasurement(cfg, model)
      for micro_step, micro_batch in enumerate(trainloader, 1):
          with throughput:
              train_loss = engine.step(micro_batch)
          ...
          if is_step:
              metrics = throughput.get_metrics()
  """

  def __init__(self, cfg, model, non_emb_params):
    self.ignore_steps = 20
    self.measure_steps = 20
    self.cfg = cfg
    self.model = model
    self.gas = cfg.grad_accumulation_steps
    self.non_emb_params = non_emb_params
    self.mbs = cfg.micro_batch_size
    self.seq_len = cfg.seq_len

    self._micro_step = 0
    self._current_step_time = 0.0
    self._measured_step_times = []
    self._flops_per_step = 0
    self._start_time = None
    self._flop_counter = None
    self._flop_exit = None

  def _training_step(self):
    """Return the 1-indexed training step the current micro-step belongs to."""
    return (self._micro_step - 1) // self.gas + 1

  def _is_first_micro_step_of_step(self):
    """True iff the current micro-step is the first of its training step."""
    return (self._micro_step - 1) % self.gas == 0

  def _is_step_boundary(self):
    """True iff the current micro-step is the last of its training step."""
    return self._micro_step % self.gas == 0

  def __enter__(self):
    self._micro_step += 1
    step = self._training_step()

    if step <= self.ignore_steps + self.measure_steps:
      torch.cuda.synchronize()
      self._start_time = time.perf_counter()

    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    step = self._training_step()
    if self.ignore_steps < step <= self.ignore_steps + self.measure_steps:
      # Capture FLOPs once, on the first micro-step of the first
      # measurement step, then scale to the full step.
      if step == self.ignore_steps + 1 and self._is_first_micro_step_of_step():
        tokens_per_step = self.mbs * self.seq_len * self.gas
        self._flops_per_step = 6 * self.non_emb_params * tokens_per_step

      torch.cuda.synchronize()
      end = time.perf_counter()
      self._current_step_time += end - self._start_time

      # When a step boundary is reached, finalise that step's time.
      if self._is_step_boundary():
        self._measured_step_times.append(self._current_step_time)
        self._current_step_time = 0.0

    return False

  def get_metrics(self):
    """Return averaged ``(step_time, flops_per_step)``, or ``None``
    if the measurement window hasn't finished yet."""
    if len(self._measured_step_times) < self.measure_steps:
      return None
    avg_step_time = sum(self._measured_step_times) / self.measure_steps
    return (avg_step_time, self._flops_per_step)


def parse_throughput_metrics(throughput_metrics, cfg, world_size):
  step_time, flops_per_step = throughput_metrics
  tokens_this_step = cfg.micro_batch_size * cfg.seq_len * cfg.grad_accumulation_steps * world_size
  tokens_per_sec = tokens_this_step / step_time

  gpu_name = torch.cuda.get_device_name(0)
  gpu_peak_flops_per_sec = GPU_PEAK_FLOPS_PER_SEC_MAP.get(gpu_name, None)
  if gpu_peak_flops_per_sec is not None:
    mfu = (world_size * flops_per_step / step_time) / (world_size * gpu_peak_flops_per_sec)

  throughput_logs = {
    'throughput/world_size': world_size,
    'throughput/step_time': step_time,
    'throughput/tokens_per_sec': tokens_per_sec,
    'throughput/mfu': mfu,
    'throughput/flops_per_step': world_size * flops_per_step,
  }
  return throughput_logs


"""
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


"""
